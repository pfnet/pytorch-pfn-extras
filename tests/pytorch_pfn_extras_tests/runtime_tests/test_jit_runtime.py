import types
import warnings

import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.utils.comparer as _comp
import torch
import torch.nn.functional as F
from pytorch_pfn_extras.onnx._as_output import trace


class JITRuntime(ppe.runtime.PyTorchRuntime):
    def move_module(self, module):
        def new_forward(self, *args):
            if hasattr(self, "_traced_mod"):
                out = self._traced_mod(*args)
                inter_size = len(self._names)
                if inter_size == 0:
                    return out
                if not isinstance(out, (tuple, list)):
                    out = [out]
                return dict(
                    **{str(i): x for i, x in enumerate(out[:-inter_size])},
                    **{
                        name: x
                        for name, x in zip(self._names, out[-inter_size:])
                    },
                )

            new_forward = self.forward
            self.forward = self._forward_with_init

            with trace(self) as (new_module, outputs):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._traced_mod = torch.jit.trace_module(
                        new_module, {"forward": args}
                    )
                self._names = [out.name for out in outputs.values]

            self.forward = new_forward
            return self.forward(*args)

        def forward_with_init(self, *args, **kwargs):
            # `module.forward` is called multiple times while tracing.
            handler = getattr(_comp._thread_local, "handler", None)
            if handler is not None:
                handler._reset_intermediate_values()
            return self._orig_forward(*args, **kwargs)

        module._orig_forward = module.forward
        module._forward_with_init = types.MethodType(forward_with_init, module)
        module.forward = types.MethodType(new_forward, module)

        def new_state_dict(self, *args, **kwargs):
            return self._traced_mod._ppe_as_out_module.state_dict()

        module.state_dict = types.MethodType(new_state_dict, module)
        return module

    def move_tensor(self, tensor):
        return tensor


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(20, 10)

    def forward(self, x, t):
        y = self.model(x)
        prefix = "train" if self.training else "val"
        loss = F.l1_loss(y, t)
        ppe.reporting.report({prefix + "/loss": loss})
        return loss


def _get_jit_cpu_model(device_type):
    model = MyModel()
    ppe.runtime.runtime_registry.register(device_type, JITRuntime)
    ppe.to(model, device_type)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def test_jit_runtime_trainer():
    model, optimizer = _get_jit_cpu_model("jit-cpu")
    trainer = ppe.engine.create_trainer(model, optimizer, 10, device="jit-cpu")
    data = torch.utils.data.DataLoader(
        [
            (
                torch.rand(
                    20,
                ),
                torch.rand(
                    10,
                ),
            )
            for i in range(100)
        ]
    )
    trainer.run(data)


def test_jit_runtime_evaluator():
    model, optimizer = _get_jit_cpu_model("jit-cpu")
    evaluator = ppe.engine.create_evaluator(model, device="jit-cpu")
    data = torch.utils.data.DataLoader(
        [
            (
                torch.rand(
                    20,
                ),
                torch.rand(
                    10,
                ),
            )
            for i in range(100)
        ]
    )
    evaluator.run(data)


def test_jit_runtime_trainer_with_evaluator():
    model, optimizer = _get_jit_cpu_model("jit-cpu")
    evaluator = ppe.engine.create_evaluator(model, device="jit-cpu")
    trainer = ppe.engine.create_trainer(
        model, optimizer, 10, device="jit-cpu", evaluator=evaluator
    )
    data = torch.utils.data.DataLoader(
        [
            (
                torch.rand(
                    20,
                ),
                torch.rand(
                    10,
                ),
            )
            for i in range(100)
        ]
    )
    trainer.run(data, data)
