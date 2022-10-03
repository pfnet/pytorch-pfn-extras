from typing import Any


def export(*args: Any, **kwargs: Any) -> Any: ...  # NOQA
def export_testcase(*args: Any, **kwargs: Any) -> Any: ...  # NOQA
def is_large_tensor(*args: Any, **kwargs: Any) -> Any: ...  # NOQA
def annotate(*args: Any, **kwargs: Any) -> Any: ...  # NOQA
def apply_annotation(*args: Any, **kwargs: Any) -> Any: ...  # NOQA
def scoped_anchor(*args: Any, **kwargs: Any) -> Any: ...  # NOQA
def as_output(*args: Any, **kwargs: Any) -> Any: ...  # NOQA
def grad(*args: Any, **kwargs: Any) -> Any: ...  # NOQA
def load_model(*args: Any, **kwargs: Any) -> Any: ...  # NOQA


LARGE_TENSOR_DATA_THRESHOLD: int
available: bool
