from typing import Any, Callable, Dict, List, Optional, cast

import torch
import torch.fx
import torch.utils._pytree as pytree
from functorch.compile import make_boxed_func
from pytorch_pfn_extras._dynamo import _optimizer, _splitter
from torch._decomp import core_aten_decompositions  # type: ignore[attr-defined]
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.partitioners import _is_primal


def _dummy_bwd_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Any:
    # The bwd pass is dummy, so we just return the inputs as they are
    def run_graph(*args, **kwargs):  # type: ignore[no-untyped-def]
        return gm(*args, **kwargs)  # type: ignore[operator]

    return make_boxed_func(run_graph)


def _join_graphs(
    module_graph: torch.fx.Graph, optimizer_graph: torch.fx.Graph
) -> torch.fx.Graph:
    module_inputs: List[torch.fx.Node] = list(
        filter(_is_primal, module_graph.nodes)
    )
    module_outputs = pytree.tree_flatten(
        [node.args for node in module_graph.nodes if node.op == "output"]
    )[0]
    grads = {}
    # Fuse the two graphs
    # 1. Look for the gradients in the outputs
    for out in module_outputs:
        if out.name.startswith("grad_"):
            grads[out.name] = out

    parameters = {}
    prefix_len = len("grad_")
    for grad_name in grads:
        for inp in module_inputs:
            if grad_name[prefix_len:] == inp.name:
                parameters[inp.name] = inp

    # Look in the optimizer graph for the nodes corresponding to the gradient obtention and
    # the parameters (usually inputs) They are the `getattr` function call in the parameter
    # In place ops can be ignored at this stage and substituted later in the
    # compilation backend since they are returning the result
    opt_grad_nodes = set()
    opt_param_nodes = set()
    opt_to_model = {}
    for node in optimizer_graph.nodes:
        if node.op == "call_function" and node.target is getattr:
            if "grad" in node.args:
                opt_grad_nodes.add(node)
                # This will allow us to just add the same operations of these nodes to the real graph.
                # Note that the updates are done INPLACE so backends for custom devices need to
                # be careful
                opt_param_nodes.add(node.args[0])
                # Save a correspondence of optimizer graph to model graph
                opt_to_model[node] = grads["grad_" + node.args[0].name]
                opt_to_model[node.args[0]] = parameters[node.args[0].name]

    # Find insertion points in the graph to add the optimizer required inputs
    last_input = None
    for in_node in module_graph.nodes:
        if _is_primal(in_node):  # type: ignore[no-untyped-call]
            last_input = in_node
    # Find insertion points in the graph to add the optimizer computation
    last_node = None
    model_output_node = None
    for node in module_graph.nodes:
        if node.op == "output":
            model_output_node = node
            break
        last_node = node

    assert model_output_node is not None
    outputs = pytree.tree_flatten(model_output_node.args)[0]

    # Merge the optimizer and model graphs
    for node in optimizer_graph.nodes:
        # Skip grad obtainer
        if node.op == "call_function" and node.target is getattr:
            if "grad" in node.args:
                continue
        if _is_primal(node):  # type: ignore[no-untyped-call]
            # Add the optimizer inputs to the module inputs
            if node.name not in parameters:
                # Look the inserting point
                module_graph.inserting_after(last_input)
                new_node = module_graph.placeholder(node.name)
                opt_to_model[node] = new_node
                new_node.meta = node.meta
                last_input = new_node
            continue
        if node.op == "output":
            # Combine model and optimizer outputs
            outputs.extend(pytree.tree_flatten(node.args)[0])
            continue

        module_graph.inserting_after(last_node)
        args = tuple(
            opt_to_model[arg] if arg in opt_to_model else arg
            for arg in node.args
        )
        res = module_graph.create_node(
            node.op, node.target, args, node.kwargs, node.name
        )
        res.meta = node.meta
        opt_to_model[node] = res
        last_node = res

    # Remove the original outputs node and add the combined one
    module_graph.erase_node(model_output_node)
    module_graph.inserting_after(last_node)
    module_graph.output(outputs)
    return module_graph


def _normalize_name(name: str) -> str:
    return name.replace("param_out_", "").replace("__dot__", ".")


def _compile_module(
    module: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    user_backend: Optional[Callable[..., Any]],
    generate_backward: bool,
    decompositions: Optional[Dict[Any, Callable]],
) -> Callable[..., Any]:
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module needs to be a torch.nn.Module instance")

    names = []
    parameters_and_buffers: List[torch.Tensor] = []

    def _graph_getter(gm, inputs):  # type: ignore[no-untyped-def]
        parameters_optimizer = []
        state_optimizer = []
        # TODO(ecastill) call the optimizer compiler here!
        if optimizer is not None:
            opt_graph, opt_outputs = _optimizer._compile_optimizer(
                module, optimizer
            )
            # gm.graph is modified in place with the added optimizer steps
            _join_graphs(gm.graph, opt_graph)
            n_opt_outs = len(opt_outputs)
            for node in opt_outputs:
                for n, p in module.named_parameters():
                    if _normalize_name(node.name) == n:
                        parameters_optimizer.append(p)

            for _, p in module.named_parameters():
                for p_n in optimizer.state[p]:  # type: ignore[index]
                    state_tensor = optimizer.state[p][p_n]  # type: ignore[index]
                    if state_tensor is not None:
                        state_optimizer.append(state_tensor)

        # Create the function that deals with the optimizer outputs
        # TODO(set this as arg)
        supports_inplace = True
        gm.recompile()  # Sync the module to the graph changes
        if user_backend is None:
            func = gm
        else:
            func = user_backend(gm, inputs)
            supports_inplace = False
        n_params = len(parameters_optimizer)

        def _model_opt_func(*args, **kwargs):  # type: ignore[no-untyped-def]
            # Need to retrieve the optimizer state and concat it to the
            # arguments
            outs = func(*(args + tuple(state_optimizer)), **kwargs)
            # Iterate the returned parameters and copy them into the
            # Model real ones (sync)
            if optimizer is not None:
                opt_outs = outs[-n_opt_outs:]
                if not supports_inplace:
                    for i in range(n_opt_outs):
                        if i < n_params:
                            parameters_optimizer[i].data.copy_(opt_outs[i])
                        else:
                            state_optimizer[i - n_params].data.copy_(
                                opt_outs[i]
                            )
                return outs[:n_opt_outs]
            return outs

        return make_boxed_func(_model_opt_func)

    # These are the first arguments passed to the functions
    # They will be the names of the inputs, replacing the primals
    # Extract the parameters name that the graph will use
    for n, p in module.named_parameters():
        parameters_and_buffers.append(p)
        names.append(n)

    for n, b in module.named_buffers():
        parameters_and_buffers.append(b)
        names.append(n)

    # This may be to simplistic ..., would be better to set a `mode`?
    partitioner: _splitter._Splitter
    if generate_backward:
        partitioner = _splitter.JointGraph(names)
    else:
        partitioner = _splitter.ForwardOnly(names)

    aot_backend = aot_autograd(  # type: ignore[no-untyped-call]
        fw_compiler=_graph_getter,
        bw_compiler=_dummy_bwd_backend,
        partition_fn=partitioner.partition,
        decompositions=decompositions,
    )
    module_opt = torch.compile(module, fullgraph=True, backend=aot_backend)  # type: ignore[attr-defined]
    return cast(Callable[..., Any], module_opt)  # type: ignore[redundant-cast]


def compile(
    module: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    backend: Optional[Callable[..., Any]] = None,
    *,
    generate_backward: bool = True,
    decompositions: Optional[Dict[Any, Callable]] = None,
) -> Callable[..., Any]:
    """Compiles a module and an optimizer in a single graph using the provided backend.

    .. note::
        The backend object needs to be a callable accepting a ``torch.fx.GraphModule``
        and a list of ``torch.Tensor`` and return a ``Callable`` as specified by
        https://pytorch.org/docs/2.0/dynamo/custom-backends.html#custom-backends

    .. note::
        Modules that are split in multiple graphs are not supported. ``torch.compiled``
        is called with the ``fullgraph=True`` argument.

    Args:
        module:
            torch.nn.Module to be compiled
        optimizer:
            Optimizer object associated to the module. It will be traced and its
            operations included in the module graph. Some dry run operations
            may be performed to fully initialize the optimizer status.
        backend (optional):
            Object to process the graph and compile it for custom devices, will
            use PyTorch dynamo by default if not specified.
        generate_backward:
            Add the backward pass to the graph. Default is ``True``.
        decompositions (optional):
            Custom mapping for decompose a torch op into simple ops. Default is
            ``None`` and resorts to `torch._decomp.core_aten_decompositions()`
    """

    if decompositions is None:
        decompositions = core_aten_decompositions()

    module_opt = _compile_module(
        module,
        optimizer,
        backend,
        generate_backward,
        decompositions,
    )
    return module_opt
