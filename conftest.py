from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from danling.tensors import NestedTensor


def pytest_make_parametrize_id(config: pytest.Config, val: object, argname: str) -> str | None:
    r"""Give common parameter values stable, readable collection ids."""

    del config, argname
    if isinstance(val, torch.dtype):
        return str(val).removeprefix("torch.")
    if isinstance(val, torch.device):
        return str(val)
    if isinstance(val, type) and issubclass(val, BaseException):
        return val.__name__
    return None


@pytest.fixture
def cuda(request: pytest.FixtureRequest) -> torch.device:
    r"""Return the CUDA device, or skip a test that requires one."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    marker = request.node.get_closest_marker("cuda_device")
    if marker is not None:
        index = marker.args[0]
        if index >= torch.cuda.device_count():
            pytest.skip(f"CUDA device {index} is not available")
        return torch.device("cuda", index)
    return torch.device("cuda")


@pytest.fixture
def device(request: pytest.FixtureRequest) -> torch.device:
    r"""Resolve a CPU/CUDA parameter using the case's CUDA device selection."""

    selected = torch.device(request.param)
    return request.getfixturevalue("cuda") if selected.type == "cuda" else selected


@pytest.fixture
def compiler_cache():
    """Isolate compiled-callable caches and release them even when a test fails."""

    torch.compiler.reset()
    try:
        yield
    finally:
        torch.compiler.reset()


@pytest.fixture(autouse=True)
def pin_float32_matmul_precision():
    r"""Pin CUDA matmul precision to IEEE for each test, restoring the previous state afterward."""

    original_generic = torch.backends.fp32_precision  # type: ignore[attr-defined]
    original_backend = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.fp32_precision = original_generic  # type: ignore[attr-defined]
        torch.backends.cuda.matmul.fp32_precision = original_backend


@pytest.fixture
def logical_negative_view():
    def view(value):
        if isinstance(value, NestedTensor):
            return value.packed_like(torch._neg_view(-value.concat))
        return torch._neg_view(-value)

    return view


@pytest.fixture
def saved_strided_hook():
    def unpack(value):
        if isinstance(value, NestedTensor):
            return value.packed_like(unpack(value.concat))
        storage = value.new_empty(value.numel() * 2)
        view = storage[::2].reshape(value.shape)
        view.copy_(value)
        return view

    @contextmanager
    def hooks():
        with torch.autograd.graph.saved_tensors_hooks(lambda value: value.detach(), unpack):
            yield

    return hooks
