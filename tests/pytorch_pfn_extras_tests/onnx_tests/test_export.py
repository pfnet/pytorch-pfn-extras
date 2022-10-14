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


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
@pytest.mark.filterwarnings("ignore:Exporting a model to ONNX with a batch_size other than 1.*:UserWarning")
def test_nested():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.LSTM(13, 17, 2)
            self.x = torch.nn.parameter.Parameter(torch.randn(3, 7, 13))

        def forward(self, *hidden):
            return self.rnn(self.x, tuple(hidden))

    m = run_model_test(
        Model(), (torch.randn(2, 7, 17), torch.randn(2, 7, 17)),
        skip_oxrt=True, output_names=["a", "b", "c"])
