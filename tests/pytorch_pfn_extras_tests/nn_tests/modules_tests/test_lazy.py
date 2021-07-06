import tempfile

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pytorch_pfn_extras.nn.modules.lazy import LazyInitializationMixin
from pytorch_pfn_extras.nn.modules.lazy import UninitializedParameter


class _MyFunc(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('const', torch.full((in_features,), 1.0))
        self._reset_params()

    def forward(self, input):
        return F.linear(input + self.const, self.weight)

    def _reset_params(self):
        self.weight.data.uniform_(-0.1, 0.1)


class _LazyMyFunc(LazyInitializationMixin, _MyFunc):

    lazy_parameter_names = ('weight',)
    lazy_buffer_names = ('const',)

    def __init__(self, in_features, out_features):
        super().__init__(in_features or 0, out_features)
        if in_features is None:
            self.in_features = None
            self.weight = UninitializedParameter()

    def forward(self, input):
        if isinstance(self.weight, UninitializedParameter):
            self.in_features = input.shape[-1]
            self.weight = torch.nn.Parameter(
                self.weight.new_empty((self.out_features, self.in_features)))
            self.const = self.const.new_full((self.in_features,), 1)
            self._reset_params()
            self.to(input.device)
        return super().forward(input)

    def _reset_params(self):
        if self.lazy_parmeters_determined:
            super()._reset_params()


class LazyTestBase:

    def get_original_module(self):
        raise NotImplementedError

    def get_lazy_module(self):
        raise NotImplementedError

    def get_input(self):
        raise NotImplementedError

    def test_basic(self):
        torch.manual_seed(0)
        input = self.get_input()

        torch.manual_seed(0)
        orig_module = self.get_original_module()
        expected = orig_module(input)

        lazy_module = self.get_lazy_module()
        torch.manual_seed(0)
        actual = lazy_module(input)

        assert expected.shape == actual.shape
        assert (expected == actual).all()

    @pytest.mark.gpu()
    def test_cuda(self):
        torch.manual_seed(0)
        input = self.get_input().cuda()

        m1 = self.get_lazy_module()
        m1.cuda()
        torch.manual_seed(0)
        expected = m1(input)

        m2 = self.get_lazy_module()
        m2.cuda()
        torch.manual_seed(0)
        actual = m2(input)

        assert expected.shape == actual.shape
        assert (expected == actual).all()

    def test_share_memory(self):
        m1 = self.get_lazy_module()
        with pytest.raises(RuntimeError):
            m1.share_memory()

    def test_double(self):
        torch.manual_seed(0)
        input = self.get_input().double()

        m1 = self.get_lazy_module()
        m1.double()
        torch.manual_seed(0)
        expected = m1(input)

        m2 = self.get_lazy_module()
        m2.double()
        torch.manual_seed(0)
        actual = m2(input)

        assert expected.shape == actual.shape
        assert (expected == actual).all()

    def test_lazy_warning(self):
        m = self.get_lazy_module()
        with pytest.warns(UserWarning) as record:
            torch.optim.SGD(m.parameters(), lr=0.1)
        assert ('Use of uninitialized lazy parameter in Optimizer '
                'has been detected' in record[0].message.args[0])

    @pytest.mark.parametrize('init_src, init_dst', [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ])
    def test_save_load(self, init_src, init_dst):
        torch.manual_seed(0)
        input = self.get_input()
        model_src = self.get_lazy_module()
        model_dst = self.get_lazy_module()
        if init_src:
            torch.manual_seed(0)
            model_src(input)

        if init_dst:
            torch.manual_seed(0)
            model_dst(input)
            module_params = [
                getattr(model_dst, name)
                for name in model_dst.lazy_parameter_names
            ]
            module_buffers = [
                getattr(model_dst, name)
                for name in model_dst.lazy_buffer_names
            ]

        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(model_src.state_dict(), f.name)
            if not init_src and init_dst:
                with pytest.raises(RuntimeError):
                    model_dst.load_state_dict(torch.load(f.name))
            else:
                model_dst.load_state_dict(torch.load(f.name))

        # Ensure that if the model was initialized, the parameters are the same
        # after loading a state dict
        if init_dst:
            for name, param in zip(
                model_dst.lazy_parameter_names, module_params
            ):
                assert getattr(model_dst, name).data_ptr() == param.data_ptr()
            for name, buffer in zip(
                model_dst.lazy_buffer_names, module_buffers
            ):
                assert getattr(model_dst, name).data_ptr() == buffer.data_ptr()

        torch.manual_seed(0)
        expected = model_src(input)
        torch.manual_seed(0)
        actual = model_dst(input)

        assert (expected == actual).all()


class TestLazyMyFunc(LazyTestBase):

    def get_original_module(self):
        return _MyFunc(10, 20)

    def get_lazy_module(self):
        return _LazyMyFunc(None, 20)

    def get_input(self):
        return torch.rand(20, 10)
