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
@pytest.mark.parametrize('k', [4, 16])
def test_mfma_f64_layout(hip_backend, compiler, k):
    hip = hip_backend.hip
    rng = np.random.default_rng(k)
    a, b = rng.standard_normal((16, k)), rng.standard_normal((k, 16))
    d = np.zeros((16, 16))

    bufs = [hip.mem_alloc(x.nbytes) for x in (a, b, d)]
    for buf, x in zip(bufs, (a, b)):
        hip.memcpy(buf, x, x.nbytes)

    mod = RawHIPModule(hip_backend, HIP_V2 / 'mfma_test.hip',
                       HIP_V2 / 'include', compiler=compiler)
    kern = mod.kernel('mfma_gemm', [np.int32, PTR, PTR, PTR], (1, 1, 1),
                      (64, 1, 1))
    kern.set_args(k, *bufs)
    hip_backend.run_kernels([kern], wait=True)
    hip.memcpy(d, bufs[2], d.nbytes)

    assert np.allclose(d, a @ b, rtol=0, atol=1e-13)
