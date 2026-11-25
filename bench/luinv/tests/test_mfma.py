from pathlib import Path

import numpy as np
import pytest

from bench.luinv.problem import make_backend
from bench.luinv.rawhip import PTR, RawHIPModule


HIP_V2 = Path(__file__).parents[1] / 'variants' / 'hip_v2'


@pytest.fixture
def hip_backend(backend_name):
    if backend_name != 'hip':
        pytest.skip('MFMA needs the HIP backend')

    backend = make_backend('hip')
    if not backend.props['gcn_arch_name'].startswith(('gfx90a', 'gfx94')):
        pytest.skip('no fp64 MFMA on this GPU')

    return backend


@pytest.mark.parametrize('compiler', ['hiprtc', 'hipcc'])
@pytest.mark.parametrize('dtype', ['f64', 'f32'])
@pytest.mark.parametrize('k', [4, 16])
def test_mfma_layout(hip_backend, compiler, dtype, k):
    hip = hip_backend.hip
    rng = np.random.default_rng(k)
    dt = np.float64 if dtype == 'f64' else np.float32
    a = rng.standard_normal((16, k)).astype(dt)
    b = rng.standard_normal((k, 16)).astype(dt)
    d = np.zeros((16, 16), dtype=dt)

    bufs = [hip.mem_alloc(x.nbytes) for x in (a, b, d)]
    for buf, x in zip(bufs, (a, b)):
        hip.memcpy(buf, x, x.nbytes)

    mod = RawHIPModule(hip_backend, HIP_V2 / 'mfma_test.hip',
                       HIP_V2 / 'include', compiler=compiler)
    kern = mod.kernel(f'mfma_gemm_{dtype}', [np.int32, PTR, PTR, PTR],
                      (1, 1, 1), (64, 1, 1))
    kern.set_args(k, *bufs)
    hip_backend.run_kernels([kern], wait=True)
    hip.memcpy(d, bufs[2], d.nbytes)

    ref = a.astype(np.float64) @ b.astype(np.float64)
    assert np.allclose(d, ref, rtol=0, atol=1e-13 if dtype == 'f64' else 1e-5)
