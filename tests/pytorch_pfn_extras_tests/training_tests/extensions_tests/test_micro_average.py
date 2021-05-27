import numpy

import pytorch_pfn_extras as ppe


def test_run():
    trigger_iters = 3
    data_shape = (4, trigger_iters)
    data_total = numpy.random.randint(7, 32, size=data_shape)

    # NumPy<1.17 does not support array-like inputs in `numpy.random.randint`.
    data_correct = numpy.random.randint(10000, size=data_shape) % data_total

    manager = ppe.training.ExtensionsManager({}, [], 100, iters_per_epoch=5)

    extension = ppe.training.extensions.MicroAverage(
        "main/correct",
        "main/total",
        "main/accuracy",
        (trigger_iters, "iteration"),
    )
    manager.extend(extension, (1, "iteration"))

    for js in numpy.ndindex(data_shape):
        with manager.run_iteration():
            ppe.reporting.report(
                {
                    "main/correct": data_correct[js],
                    "main/total": data_total[js],
                }
            )
        assert (
            # average is computed every trigger_iters
            ("main/accuracy" in manager.observation)
            == (js[1] == trigger_iters - 1)
        )
