import platform

import pytest


@pytest.fixture(scope="session")
def backend():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_spec_bench.models import initialize

    metal = (platform.system(), platform.machine()) == ("Darwin", "arm64")
    mx = initialize(require_metal=metal)
    if not metal:
        # The CPU development check works without the host C++ JIT toolchain.
        mx.disable_compile()
    return mx, metal
