from __future__ import annotations

from typing import Callable, Union

import numpy
import torch

Scalar = Union[torch.Tensor, numpy.ndarray, numpy.floating, float]
FloatLikeValue = Union[Scalar, float]
Value = Union[Scalar, Callable[[], float]]


def _as_tensor_with_reference(
    value: Scalar, reference: torch.Tensor
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(
        value, device=reference.device, dtype=reference.dtype
    )


def divide_scalar(numerator: Scalar, denominator: Scalar) -> Scalar:
    if isinstance(numerator, torch.Tensor):
        denominator_as_tensor = _as_tensor_with_reference(
            denominator, numerator
        )
        return numerator / denominator_as_tensor
    if isinstance(denominator, torch.Tensor):
        numerator_as_tensor = _as_tensor_with_reference(numerator, denominator)
        return numerator_as_tensor / denominator
    return numerator / denominator


def multiply_scalar(left: Scalar, right: Scalar) -> Scalar:
    if isinstance(left, torch.Tensor):
        right_as_tensor = _as_tensor_with_reference(right, left)
        return left * right_as_tensor
    if isinstance(right, torch.Tensor):
        left_as_tensor = _as_tensor_with_reference(left, right)
        return left_as_tensor * right
    return left * right


def subtract_scalar(minuend: Scalar, subtrahend: Scalar) -> Scalar:
    if isinstance(minuend, torch.Tensor):
        subtrahend_as_tensor = _as_tensor_with_reference(subtrahend, minuend)
        return minuend - subtrahend_as_tensor
    if isinstance(subtrahend, torch.Tensor):
        minuend_as_tensor = _as_tensor_with_reference(minuend, subtrahend)
        return minuend_as_tensor - subtrahend
    return minuend - subtrahend
