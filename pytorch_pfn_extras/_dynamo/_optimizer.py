import contextlib
import types
from typing import Any, Dict, Generator, List, Tuple

import pytorch_pfn_extras
import torch
import torch.fx

if pytorch_pfn_extras.requires("2.5"):
    unset_fake_temporarily = (
        torch._subclasses.fake_tensor.unset_fake_temporarily  # type: ignore[attr-defined]
    )
else:
    unset_fake_temporarily = (
        torch.fx.experimental.proxy_tensor.maybe_disable_fake_tensor_mode  # type: ignore[attr-defined]
    )


# patch the torch.optim.SGD._init_group function to avoid the
# symbolically traced variables cannot be used as inputs to control flow error
# by replacing this function in SGD optimizer instances
def _sgd_init_group(  # type: ignore[no-untyped-def]
    self, group, params_with_grad, d_p_list, momentum_buffer_list
):
    has_sparse_grad = False

    for p in group["params"]:
        if p.grad is not None:
            params_with_grad.append(p)
            d_p_list.append(p.grad)
            # if p.grad.is_sparse:
            #     has_sparse_grad = True
            has_sparse_grad = p.grad.is_sparse

            state = self.state[p]
            if "momentum_buffer" not in state:
                momentum_buffer_list.append(None)
            else:
                momentum_buffer_list.append(state["momentum_buffer"])

    return has_sparse_grad


def _get_parameters_names(
    model: torch.nn.Module,
) -> Dict[torch.Tensor, str]:
    param_names: Dict[torch.Tensor, str] = {}
    for n, p in model.named_parameters():
        param_names[p] = n
    return param_names


@contextlib.contextmanager
def _initialize_optimizer(
    optimizer: torch.optim.Optimizer, module: torch.nn.Module
) -> Generator[Dict[torch.Tensor, str], None, None]:
    if isinstance(optimizer, torch.optim.SGD):
        optimizer._init_group = types.MethodType(_sgd_init_group, optimizer)  # type: ignore[attr-defined,method-assign]

    # Replace the optimizer parameters with zero tensors so that the step functions
    # will initialize the state but doesn"t modify the module real weights
    # param_to_dummy = {}  # Keeps references so that `state` tensors can be reset
    param_groups: List[List[torch.Tensor]] = []
    param_to_dummy: Dict[torch.Tensor, torch.Tensor] = {}
    with unset_fake_temporarily():  # type: ignore[attr-defined,no-untyped-call]
        names = _get_parameters_names(module)

        for p_group in optimizer.param_groups:
            param_groups.append([])
            for i, param in enumerate(p_group["params"]):
                dummy = torch.zeros_like(param)
                dummy.grad = torch.zeros_like(param)
                param_groups[-1].append(param)
                names[dummy] = names[param]
                param_to_dummy[param] = dummy
                p_group["params"][i] = dummy

        # This call will initialize the `.state` values so fx can trace its ops
        optimizer.step()

    yield names

    with unset_fake_temporarily():  # type: ignore[attr-defined,no-untyped-call]
        # Reset the optimizer original parameters
        for i, p_group in enumerate(optimizer.param_groups):
            for j, _ in enumerate(p_group["params"]):
                param = param_groups[i][j]
                p_group["params"][j] = param
                dummy = param_to_dummy[param]
                optimizer.state[param] = optimizer.state[dummy]  # type: ignore[index]


def _create_meta(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "val": None,
        "tensor_meta": torch.fx.passes.shape_prop.TensorMetadata(  # type: ignore[attr-defined, arg-type]
            tensor.shape,
            tensor.dtype,
            tensor.requires_grad,
            tensor.stride(),  # type: ignore[arg-type]
            torch.preserve_format,
            tensor.data.is_quantized,
            {},
        ),
    }


def _get_shape_inference_inputs_and_metadata(
    optimizer: torch.optim.Optimizer,
) -> Tuple[Dict[torch.Tensor, Dict[str, Any]], List[torch.Tensor]]:
    params_meta = {}
    inputs = []

    with unset_fake_temporarily():  # type: ignore[attr-defined,no-untyped-call]
        for p_group in optimizer.param_groups:
            for param in p_group["params"]:
                param_tensor = param
                params_meta[param_tensor] = _create_meta(param_tensor)
                inputs.append(param_tensor)

        for p_group in optimizer.param_groups:
            for param in p_group["params"]:
                param_tensor = param
                optimizer_state: Dict[Any, Any] = optimizer.state[param_tensor]
                for state_tensor in optimizer_state.values():
                    if state_tensor is not None:
                        params_meta[state_tensor] = _create_meta(state_tensor)
                        inputs.append(state_tensor)

    return params_meta, inputs


def _create_placeholders_for_parameters_and_state(
    optimizer: torch.optim.Optimizer,
    names: Dict[torch.Tensor, str],
    opt_graph: torch.fx.Graph,
    params_meta: Dict[torch.Tensor, Dict[str, Any]],
    tracer: torch.fx.proxy.GraphAppendingTracer,
) -> Tuple[List[torch.fx.Node], List[torch.fx.Node]]:
    placeholders = []
    state = []

    params_to_proxy = {}
    for p_group in optimizer.param_groups:
        for i, param in enumerate(p_group["params"]):
            # Find param in list
            param_tensor = param
            # Dynamo uses the parameters names to create a python function
            # if special symbols in the parameter names are not replaced
            # the definition will be ill-formed and look like:
            # def forward(self, linear.weight, ...)
            # causing a syntax error
            p_name = names[param_tensor].replace(".", "__dot__")
            placeholders.append(opt_graph.placeholder(p_name))
            placeholders[-1].meta = params_meta[param_tensor]
            proxy = torch.fx.Proxy(placeholders[i], tracer)
            optimizer.state[proxy] = optimizer.state[param_tensor].copy()  # type: ignore[index]
            params_to_proxy[param_tensor] = proxy

    for p_group in optimizer.param_groups:
        for i, param in enumerate(p_group["params"]):
            # Find param in list
            # May need to replace `.` with `@`
            param_tensor = param
            p_name = names[param_tensor].replace(".", "__dot__")
            proxy = params_to_proxy[param_tensor]
            for p in optimizer.state[proxy]:  # type: ignore[index]
                state_tensor = optimizer.state[param_tensor][p]
                if state_tensor is not None:
                    state.append(opt_graph.placeholder(f"state_{p}_{p_name}"))
                    optimizer.state[proxy][p] = torch.fx.Proxy(  # type: ignore[index]
                        state[-1], tracer
                    )
                    state[-1].meta = params_meta[state_tensor]
            p_group["params"][i] = proxy

    return placeholders, state


def _is_inplace(node: torch.fx.Node, arg: torch.fx.Node) -> bool:
    # There is no easy way to detect inplace ops in torch, but they are
    # defined as tensor methods with a "_" suffix. ("add_", "mul_")
    return (
        node.op == "call_method"  # type: ignore[return-value]
        and node.args[0] == arg
        and node.target[-1] == "_"  # type: ignore[index]
    )


def _get_last_inplace_update(
    opt_graph: torch.fx.Graph,
) -> Dict[torch.fx.Node, torch.fx.Node]:
    last_inplace = {}
    for node in opt_graph.nodes:
        last_node = node
        for o_node in opt_graph.nodes:
            # If its an inplace modifying op, then its likely to be the update
            if _is_inplace(o_node, last_node):
                last_node = o_node
                last_inplace[node] = o_node

    return last_inplace


def _adjust_inplace_ops(
    opt_graph: torch.fx.Graph, last_inplace: Dict[torch.fx.Node, torch.fx.Node]
) -> None:
    # This is to avoid cases such as:
    # b = a.add_()  # a is modified in place and returns the value
    #               # a=b with dynamo, but it is possible to be a!=b in other backends
    # c = torch.exp(a)  # if in-place is supported this is correct, but if not we want to be torch.exp(b)
    # This behavior is seen in momentum update for SGD
    for node in opt_graph.nodes:
        args = list(node.args)
        modified = False
        for i, a in enumerate(args):
            #  find the node that lastly modified the arg inplace before the current node
            if a in last_inplace:
                last = a
                for p_node in opt_graph.nodes:
                    if p_node == node:
                        break
                    # Identify the previous update for the current node arg
                    if _is_inplace(p_node, last):
                        last = p_node
                args[i] = last
                modified = True

        if modified:
            node.args = tuple(args)


def _compile_optimizer(
    module: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Tuple[torch.fx.Graph, List[torch.fx.Node]]:
    # Do all the optimizer crap here
    if not isinstance(module, torch.nn.Module):
        raise RuntimeError(
            "Optimizer needs module to be instance of torch.nn.Module"
        )

    opt_graph = torch.fx.Graph()
    tracer = torch.fx.proxy.GraphAppendingTracer(opt_graph)

    # Gets all the optimizer registered inputs so we can run shape inference
    with _initialize_optimizer(optimizer, module) as param_names:
        params_meta, inputs = _get_shape_inference_inputs_and_metadata(
            optimizer
        )
        (
            param_placeholders,
            state_placeholders,
        ) = _create_placeholders_for_parameters_and_state(
            optimizer, param_names, opt_graph, params_meta, tracer
        )

        # Trace the computation
        optimizer.step()
        # Look for the parameters and return their last known value
        outputs = []

        last_inplace = _get_last_inplace_update(opt_graph)

        # The last inplace update will be the optimizer outputs
        def _get_last_update(
            node_set: List[torch.fx.Node], out_prefix: str
        ) -> None:
            for node in node_set:
                last_inplace[node].name = f"{out_prefix}{node.name}"
                outputs.append(last_inplace[node])

        # Add the last node updating a parameter to the graph outputs
        _get_last_update(param_placeholders, "param_out_")
        # Add the last node updating the state (e.g. momentum) to the graph outputs
        _get_last_update(state_placeholders, "param_out_")

        # Make the nodes that have inplace ops as arguments to use the last inplace
        # update right before the node itself. in some devices, inplace updates
        # may not be real in-place ops and pytorch graph uses inplace ops nodes
        # without caring about the order
        # b = a.inplace_op()
        # c = b.inplace_op()
        # d = a + x   # Here we use a, but since the value is updated it is equivalent
        #             # to use c, and c should be used to ensure correctness.
        _adjust_inplace_ops(opt_graph, last_inplace)
        opt_graph.output(outputs)

        with unset_fake_temporarily():  # type: ignore[attr-defined,no-untyped-call]
            opt_module = torch.fx.GraphModule(torch.nn.Module(), opt_graph)
            torch.fx.passes.shape_prop.ShapeProp(opt_module).propagate(*inputs)  # type: ignore[attr-defined, no-untyped-call]

    return opt_graph, outputs
