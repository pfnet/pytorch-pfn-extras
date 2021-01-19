try:
    # Make sure that onnx is imported before importing torch in the test run.
    import onnx  # NOQA
except Exception:
    pass
