from typing import Any, List, Optional, Tuple

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._functorch.partitioners import (
    _is_primal,
    _is_tangent,
    default_partition,
)


class _Splitter:
    def partition(
        self,
        joint_module: torch.fx.GraphModule,
        _joint_inputs: Any,
        *,
        num_fwd_outputs: int,
    ) -> Tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        raise NotImplementedError("Splitters must override partition")


class JointGraph(_Splitter):
    def __init__(
        self,
        parameter_names: Optional[List[str]] = None,
    ):
        if parameter_names is not None:
            self._parameter_names = [
                n.replace(".", "__dot__") for n in parameter_names
            ]
        else:
            self._parameter_names = []

    def partition(
        self,
        joint_module: torch.fx.GraphModule,
        _joint_inputs: Any,
        *,
        num_fwd_outputs: int,
    ) -> Tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        """The calculation graph, traced in an end-to-end manner,
        of the forward-backward computation is divided into the forward graph and
        the backward graph. The forward graph includes the whole calculation process
        up until the computation of gradients. The backward graph is split as
        an identity function concerning the gradients.

        Args:
            joint_module: The end-to-end calculation graph.
            _joint_inputs: Example inputs.
            num_fwd_outputs: Number of forward outputs.
        Returns:
            The two returned GraphModule objects must adhere to the following interface:
            - forward_graph_module: ((*primal_inputs) -> any_outputs)
            - backward_graph_module: ((subset_of_forward_outputs, *tangent_inputs) -> parameter_grads)
        """
        primal_inputs: List[torch.fx.Node] = list(
            filter(_is_primal, joint_module.graph.nodes)
        )
        tangent_inputs: List[torch.fx.Node] = list(
            filter(_is_tangent, joint_module.graph.nodes)
        )
        outputs = pytree.tree_flatten(
            [
                node.args
                for node in joint_module.graph.nodes
                if node.op == "output"
            ]
        )[0]
        combined_graph = torch.fx.Graph()
        env = {}
        for i, node in enumerate(primal_inputs):
            new_node = combined_graph.placeholder(
                self._parameter_names[i]
                if i < len(self._parameter_names)
                else f"input_{i - len(self._parameter_names)}"
            )
            new_node.meta = node.meta
            env[node] = new_node
        # The tangents will be transformed to constant ops
        # Depending on the module this is different :(
        # Maybe we can retrieve shape?
        for node in tangent_inputs:
            new_node = combined_graph.call_function(
                torch.ops.aten.ones, args=(node.meta.get("tensor_meta").shape,)  # type: ignore[union-attr]
            )
            new_node.meta = node.meta
            env[node] = new_node

        assert len(tangent_inputs) == num_fwd_outputs

        for node in joint_module.graph.nodes:
            if node in primal_inputs or node in tangent_inputs:
                continue
            if node.op != "output":
                env[node] = combined_graph.node_copy(node, lambda x: env[x])

        outs = set()
        combined_outputs = []
        # Some outputs are repeated, just return them only once
        # We use `env[node]` to use newly created nodes for tangent if we are
        # going to return them
        for node in outputs:
            if node is None:
                # This is the case where the corresponding input doesn"t need a grad
                continue  # type: ignore[unreachable]
            if env[node] not in outs:
                combined_outputs.append(env[node])
                outs.add(env[node])

        for i, node in enumerate(combined_outputs[:num_fwd_outputs]):
            node.name = f"fwd_out_{i}"
        for i, node in enumerate(combined_outputs[num_fwd_outputs:]):
            if node is not None:
                node.name = "grad_" + (
                    self._parameter_names[i]
                    if i < len(self._parameter_names)
                    else f"input_{i - len(self._parameter_names)}"
                )

        combined_graph.output(combined_outputs)
        fwd_module = torch.fx.GraphModule(joint_module, combined_graph)
        # Now we create a graph for backward that is just the identities of the original inputs
        # Since they are now known
        bwd_graph = torch.fx.Graph()
        bwd_outputs = outputs[num_fwd_outputs:]
        out_nodes = []
        env = {}

        for node in bwd_outputs:
            # Rename it
            if node is None:
                out_nodes.append(node)  # type: ignore[unreachable]
                continue
            if node not in env:
                new_node = bwd_graph.placeholder(node.name)
                if node not in tangent_inputs:
                    env[node] = new_node
                new_node.meta = node.meta
            out_nodes.append(new_node)

        for node in tangent_inputs:
            if node not in env:
                new_node = bwd_graph.placeholder(node.name)
                new_node.meta = node.meta
                env[node] = new_node

        bwd_graph.output(out_nodes)
        bwd_module = torch.fx.GraphModule(joint_module, bwd_graph)
        return (fwd_module, bwd_module)


class ForwardOnly(_Splitter):
    def __init__(
        self,
        parameter_names: Optional[List[str]] = None,
    ):
        if parameter_names is not None:
            self._parameter_names = [
                n.replace(".", "__dot__") for n in parameter_names
            ]
        else:
            self._parameter_names = []

    def partition(
        self,
        joint_module: torch.fx.GraphModule,
        _joint_inputs: Any,
        *,
        num_fwd_outputs: int,
    ) -> Tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        fwd_module, _ = default_partition(
            joint_module,
            _joint_inputs,
            num_fwd_outputs=num_fwd_outputs,
        )
        fwd_graph = fwd_module.graph
        # Change the input names in the fwd graph
        primal_inputs: List[torch.fx.Node] = list(
            filter(_is_primal, fwd_graph.nodes)
        )
        for i, node in enumerate(primal_inputs):
            node.name = (
                self._parameter_names[i]
                if i < len(self._parameter_names)
                else f"input_{i - len(self._parameter_names)}"
            )

        # The joint graph has the forward and backward outputs together as output values
        # by accessing the output node of the graph (there is only one output node)
        # the node has a list of all the variables the graph returns in the node.args
        # https://pytorch.org/docs/stable/fx.html#a-quick-primer-on-graphs
        output_node = [node for node in fwd_graph.nodes if node.op == "output"][
            0
        ]
        outputs = pytree.tree_flatten(output_node.args)[0]
        fwd_graph.erase_node(output_node)
        # Select only the return values from the forward pass
        fwd_graph.output(outputs[:num_fwd_outputs])
        fwd_module = torch.fx.GraphModule(joint_module, fwd_graph)

        # We now create a dummy graph that returns the outputs of the backward pass
        # Notice that the graph needs to return as many values as the inputs of the
        # forward pass. The outputs of the joint graph returns additional values
        # besides the gradients.
        bwd_graph = torch.fx.Graph()
        # Needs to create one gradient per each input element
        bwd_outs = []
        for i_node in primal_inputs:
            bwd_outs.append(
                bwd_graph.call_function(
                    torch.zeros, (i_node.meta.get("tensor_meta").shape,)
                )
            )
        bwd_graph.output(tuple(bwd_outs))
        bwd_module = torch.fx.GraphModule(joint_module, bwd_graph)
        return (fwd_module, bwd_module)
