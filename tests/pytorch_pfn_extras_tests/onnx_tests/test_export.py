import onnx
import pytest
import torch

from pytorch_pfn_extras_tests.onnx_tests.utils import run_model_test


def test_simple():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(20, 10)
            self._outputs = []

        def forward(self, x):
            y = self.linear(x)
            self._outputs.clear()
            self._outputs.append(y)
            return self._outputs[0]

    run_model_test(Model(), (torch.rand((20,)),))


def test_conv():
    torch.manual_seed(100)

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = torch.nn.Conv2d(1, 1, 3)

        def forward(self, x):
            return self.conv(x)

    run_model_test(Net(), (torch.rand(1, 1, 112, 112),), rtol=1e-03)


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_symbolic_function():
    class Func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            return a + 10

        @staticmethod
        def symbolic(g, a):
            return g.op(
                "Add",
                a,
                g.op("Constant", value_t=torch.tensor([10], dtype=torch.float)),
            )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return Func.apply(x) + torch.tensor([10], dtype=torch.float)

    assert hasattr(Func, "symbolic")
    run_model_test(Model(), (torch.rand((20,)),))


class AnyModel(torch.nn.Module):
    def __init__(self, fn, params):
        super(AnyModel, self).__init__()
        for name, value in params.items():
            setattr(self, name, torch.nn.parameter.Parameter(value))
        self.fn = fn

    def __call__(self, *args):
        result = self.fn(self, *args)
        return result


def test_if():
    @torch.jit.script
    def if_by_shape(x):
        if x.shape[0] == 3:
            return torch.relu(x)
        else:
            return torch.abs(x)

    run_model_test(AnyModel(lambda m, x: if_by_shape(x), {}), (torch.arange(3, dtype=torch.float),))
    run_model_test(AnyModel(lambda m, x: if_by_shape(x), {}), (torch.arange(4, dtype=torch.float),))


def test_scalar_const():
    run_model_test(AnyModel(lambda m, x: x * 3.0, {}), (torch.arange(3, dtype=torch.float),))
    run_model_test(AnyModel(lambda m, x: x * 3.0, {}), (torch.arange(3, dtype=torch.double),))


def _aranges(*shape) -> torch.Tensor:
    return torch.arange(torch.prod(torch.tensor(shape)), dtype=torch.float).reshape(*shape)


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_adaptive_max_pool():
    run_model_test(
        AnyModel(lambda m, x: torch.nn.functional.adaptive_max_pool2d(x * m.w, 1),
                 {"w": _aranges(3, 1, 1)}),
        (_aranges(2, 3, 5, 5) % 9,),
    )


def test_keep_initializers_as_inputs():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("var", torch.rand(10, 10))

        def forward(self, x):
            return self.var + x

    model: onnx.ModelProto = run_model_test(Model(), (torch.rand((10,)),), keep_initializers_as_inputs=False)
    assert len(model.graph.input) == 1
    model = run_model_test(Model(), (torch.rand((1,)),), keep_initializers_as_inputs=True)
    assert len(model.graph.input) == 2


@pytest.mark.filterwarnings("ignore:No names were found for specified dynamic axes of:UserWarning")
def test_dynamic_axes():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("var", torch.rand(10, 10))

        def forward(self, x, y):
            return self.var + x + y

    model: onnx.ModelProto = run_model_test(
        Model(), (torch.rand((10,)), torch.rand((10, 10))),
        keep_initializers_as_inputs=False,
        input_names=["x", "y"],
        output_names=["out"],
        dynamic_axes={"x": {0: "custom"}, "y": [0, 1], "out": [0]},)
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "custom"
    assert model.graph.input[1].type.tensor_type.shape.dim[0].dim_param == "y_dynamic_axes_1"
    assert model.graph.input[1].type.tensor_type.shape.dim[1].dim_param == "y_dynamic_axes_2"
    assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "out_dynamic_axes_1"


@pytest.mark.filterwarnings("ignore:No input args:UserWarning")
def test_concat():
    model: onnx.ModelProto = run_model_test(
        AnyModel(lambda m: torch.cat((m.x, m.y), axis=0),
                 {"x": _aranges(2, 3, 2), "y": _aranges(3, 3, 2)}),
        (),
    )
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Concat"
    model = run_model_test(
        AnyModel(lambda m: torch.cat((m.x, m.y), axis=1),
                 {"x": _aranges(2, 3, 2), "y": _aranges(2, 3, 2)}),
        (),
    )
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Concat"


def test_norm():
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            return torch.norm(x)

    run_model_test(Net(), (torch.rand(2, 3, 5, 7),), opset_version=13)


def test_rand():
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            return torch.rand(3) * x

    run_model_test(
        Net(), (torch.rand(2, 3),),
        opset_version=13, skip_oxrt=True)


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
@pytest.mark.filterwarnings("ignore:Exporting a model to ONNX with a batch_size other than 1.*:UserWarning")
@pytest.mark.filterwarnings("ignore:The shape inference of prim..Constant type is missing.*:UserWarning")
def test_nested():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.LSTM(13, 17, 2)
            self.x = torch.nn.parameter.Parameter(torch.randn(3, 7, 13))

        def forward(self, *hidden):
            return self.rnn(self.x, tuple(hidden))

    run_model_test(
        Model(), (torch.randn(2, 7, 17), torch.randn(2, 7, 17)),
        skip_oxrt=True, output_names=["a", "b", "c"])


@pytest.mark.filterwarnings("ignore:The shape inference of org.chainer..Add type is missing:UserWarning")
def test_custom_opsets():
    class Func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            return a + 10

        @staticmethod
        def symbolic(g, a):
            return g.op(
                "org.chainer::Add",
                a,
                g.op("Constant", value_t=torch.tensor([10], dtype=torch.float)),
            )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return Func.apply(x) + 10

    ver = 9
    m = run_model_test(
        Model(), (torch.randn(2, 7, 17),),
        skip_oxrt=True,
        custom_opsets={"org.chainer": ver})

    assert len(m.opset_import) == 2

    for o in m.opset_import:
        if o.domain == 'org.chainer':
            assert o.version == ver


def test_softmax():
    run_model_test(torch.nn.Softmax(3), (torch.randn(1, 10, 30, 30),))


def test_complex():
    class Complex(torch.nn.Module):
        def forward(self, x):
            return x + 1

    x = torch.rand(32, 32, dtype=torch.complex64)
    run_model_test(
        Complex(),
        (x,),
        check_torch_export=False,
        onnx_scalar_type_analysis=False,
        skip_oxrt=True,  # Add op in ONNX spec doesn't support complex input
    )


def test_alias_param():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(20, 20)
            self.linear2 = torch.nn.Linear(20, 20)
            self.linear2.weight = self.linear.weight
            self._outputs = []

        def forward(self, x):
            y = self.linear(x)
            y = self.linear2(y)
            self._outputs.clear()
            self._outputs.append(y)
            return self._outputs[0]

    m = run_model_test(Model(), (torch.rand((20,)),))
    params = [i.name for i in m.graph.initializer]
    assert params == ["linear2.weight", "linear.bias", "linear2.bias"]


def test_is_tracing():
    class Model(torch.nn.Module):
        def forward(self, x):
            if torch.jit.is_tracing():
                return x * x,

            ret = x * x
            return {"y": ret}

    run_model_test(Model(), (torch.rand(32, 32),))


def test_op_norm():
    import torch.onnx.symbolic_helper as sym_help

    @torch.onnx.symbolic_helper.parse_args("v", "v")
    def clamp_min(g, self, min):
        # dtype = self.type().scalarType()
        # Type info may be lost here.
        # https://github.com/pfnet/pytorch-pfn-extras/issues/578
        # min = g.op("Cast", min, to_i=sym_help.cast_pytorch_to_onnx[dtype])
        if sym_help._get_tensor_rank(min) == 0:
            max = torch.onnx.symbolic_opset9.unused(g)
            return g.op("Clip", self, min, max)
        else:
            return g.op("Max", self, min)

    @torch.onnx.symbolic_helper.parse_args("v", "v")
    def clamp_max(g, self, max):
        # dtype = self.type().scalarType()
        # Type info may be lost here.
        # https://github.com/pfnet/pytorch-pfn-extras/issues/578
        # max = g.op("Cast", max, to_i=sym_help.cast_pytorch_to_onnx[dtype])
        if sym_help._get_tensor_rank(max) == 0:
            min = torch.onnx.symbolic_opset9.unused(g)
            return g.op("Clip", self, min, max)
        else:
            return g.op("Min", self, max)

    torch.onnx.symbolic_opset11.clamp_min = clamp_min
    torch.onnx.symbolic_opset11.clamp_max = clamp_max

    @torch.onnx.symbolic_helper.parse_args("v", "v", "v")
    def clamp(g, self, min, max):
        dtype = self.type().scalarType()

        def _cast_if_not_none(tensor, dtype):
            if tensor is not None and not sym_help._is_none(tensor):
                return g.op(
                    "Cast", tensor, to_i=sym_help.cast_pytorch_to_onnx[dtype]
                )
            else:
                return tensor

        # pfto loses type info after Cast.
        # https://github.com/pfnet/pytorch-pfn-extras/issues/578
        orig_min = min
        orig_max = max

        if dtype is not None:
            min = _cast_if_not_none(min, dtype)
            max = _cast_if_not_none(max, dtype)

        if sym_help._is_none(min):
            return clamp_max(g, self, max)
        elif sym_help._is_none(max):
            return clamp_min(g, self, min)
        else:
            if (
                sym_help._get_tensor_rank(orig_min) == 0
                and sym_help._get_tensor_rank(orig_max) == 0
            ):
                return g.op("Clip", self, min, max)
            else:
                return clamp_max(g, clamp_min(g, self, min), max)

    class Clip(torch.nn.Module):
        def __init__(self):
            super(Clip, self).__init__()
            self.a = torch.rand(32, 32)

        def forward(self, x):
            return torch.clip(x, -2.0, 4.0) + torch.clip(self.a, 10, 20)

    class Proxy(torch.nn.Module):
        def __init__(self):
            super(Proxy, self).__init__()
            self.c = Clip()

        def forward(self, x):
            return self.c(x + 1)

    x = torch.rand(32, 32)
    run_model_test(
        Proxy(),
        (x,),
        do_constant_folding=False,
    )


@pytest.mark.parametrize("persistent", [True, False])
def test_persistent(persistent):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("var", torch.rand(10, 10), persistent=persistent)

        def forward(self, x):
            return self.var + x

    model: onnx.ModelProto = run_model_test(Model(), (torch.rand((10,)),), keep_initializers_as_inputs=False)
    assert len(model.graph.input) == 1
    model = run_model_test(Model(), (torch.rand((1,)),), keep_initializers_as_inputs=True)
    assert len(model.graph.input) == (2 if persistent else 1)

def test_script_device():
    @torch.jit.script
    def _select_by_mask_values(
        masks: torch.Tensor, mask_values: torch.Tensor
    ) -> torch.Tensor:
        H, W = masks.shape
        N, n_mask_value = mask_values.shape
        # TODO(take-cheeze): Using dtype=int64 instead of bool since shape inference fails
        out = torch.zeros(N, W, device=masks.device, dtype=torch.int64)
        return out
    
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return _select_by_mask_values(x, y).sum()

    run_model_test(
        Model(),
        (torch.rand(32, 32), torch.rand(32, 32)),
        input_names=["x", "y"],
        output_names=["out"],
        dynamic_axes={"x": {1: "A"}, "y": {0: "B"}},
    )
