import tempfile
import os

import pytest

import pytorch_pfn_extras as ppe


def test_tensorboard_writing():
    pytest.importorskip('tensorboard')
    data = {"a": 1, "iteration": 1}
    with tempfile.TemporaryDirectory() as tempd:
        writer = ppe.writing.TensorBoardWriter(
            out_dir=tempd, filename_suffix='_test')
        writer(None, None, data)
        # Check that the file was generated
        for snap in os.listdir(tempd):
            assert '_test' in snap
        writer.finalize()
