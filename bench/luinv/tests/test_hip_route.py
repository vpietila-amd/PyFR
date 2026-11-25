import numpy as np
import pytest

from bench.luinv import rawhip
from bench.luinv.harness import Case, build, stage_and_invert
from bench.luinv.problem import make_backend


@pytest.fixture
def hip_backend(backend_name):
    if backend_name != 'hip':
        pytest.skip('raw HIP kernels need the HIP backend')

    return make_backend('hip')


@pytest.mark.parametrize('compiler', ['hiprtc', 'hipcc'])
@pytest.mark.parametrize('n', [100, 175])
def test_hip_ref_matches_baseline(hip_backend, monkeypatch, compiler, n):
    monkeypatch.setenv('LUINV_COMPILER', compiler)

    xs = []
    for v in ('baseline', 'hip_ref'):
        case = Case(n=n, neles=130, ncolours=2, variant=v)
        prob = build(case)
        stage_and_invert(prob, case)
        xs.append(prob.read())

    assert np.array_equal(*xs)


@pytest.mark.parametrize('compiler', ['hiprtc', 'hipcc'])
def test_header_edit_rebuilds(hip_backend, tmp_path, monkeypatch, compiler):
    monkeypatch.setattr(rawhip, 'BUILD_DIR', tmp_path / 'build')
    monkeypatch.setenv('PYFR_HIP_CACHE_DIR', str(tmp_path / 'cache'))

    inc = tmp_path / 'include'
    inc.mkdir()
    entry = tmp_path / 'toy.hip'
    entry.write_text('#include "val.hpp"\n'
                     'extern "C" __global__ void toy(int *x) { *x = VAL; }\n')

    hip = hip_backend.hip
    buf = hip.mem_alloc(4)

    def run():
        mod = rawhip.RawHIPModule(hip_backend, entry, inc, compiler=compiler)
        kern = mod.kernel('toy', [rawhip.PTR], (1, 1, 1), (1, 1, 1))
        kern.set_args(buf)
        hip_backend.run_kernels([kern], wait=True)

        out = np.zeros(1, dtype=np.int32)
        hip.memcpy(out, buf, 4)
        return int(out[0])

    for val in (1, 2, 1):
        (inc / 'val.hpp').write_text(f'#define VAL {val}\n')
        assert run() == val
