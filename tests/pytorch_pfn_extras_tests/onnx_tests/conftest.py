import pytest
import torch

@pytest.fixture(scope='function', autouse=True)
def init_rand_seed():
    torch.manual_seed(100)
