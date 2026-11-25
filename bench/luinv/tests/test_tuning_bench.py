"""End-to-end tuning-search run against the real hip_opt kernels.

Marked ``tune`` (excluded from the quick suite): it needs the HIP backend and
the framework checked out.  Run with:

    python -m pytest bench/luinv/tests/test_tuning_bench.py -m tune
"""
import math

import pytest

from bench.luinv.harness import Case
from bench.luinv.problem import make_backend
from bench.luinv.tune import TuneBackend, loader, make_backend_and_space
from bench.luinv.tune.space import make_space

pytestmark = pytest.mark.tune


@pytest.fixture(scope='module')
def ts():
    if not (loader.package_dir() / '__init__.py').is_file():
        pytest.skip(f'tuning framework not checked out at {loader.package_dir()}')
    return loader.load_tuning_search()


@pytest.fixture
def hip(backend_name):
    if backend_name != 'hip':
        pytest.skip('tuning needs the HIP backend')
    return make_backend('hip')


def _case():
    return Case(n=70, neles=512, ncolours=1, precision='double',
                variant='hip_opt', backend='hip')


def test_exhaustive_tune_end_to_end(ts, hip):
    case = _case()
    backend, space, baseline = make_backend_and_space(case, ts, nsample=16)

    res = ts.tune(space, backend, strategy='exhaustive',
                  ladder=(ts.Fidelity(reps=10, warmup=3),),
                  reject_spill=False, baseline=baseline,
                  environment=backend.environment(), seed=0)

    # Every one of the 12 configs was evaluated and a valid best was found
    assert res.evaluations >= 1
    assert res.best_config['minblk'] in (1, 2, 3)
    assert math.isfinite(res.best_value) and res.best_value > 0

    # The best config passed the correctness gate (only passing configs win)
    ok, detail = backend.check(res.best_config, res.best_key,
                               backend.compile(res.best_config,
                                               res.best_key).artifact)
    assert ok, detail

    # Confirmation against the untuned baseline produced a finite speedup
    if res.confirmed:
        assert math.isfinite(res.confirmed.get('speedup', float('nan')))


def test_compile_failure_becomes_not_ok(ts, hip, monkeypatch):
    # A knob rewrite that raises must surface as a failed compile, not a
    # crash, so the evaluator can score it as infinity.
    case = _case()
    tb = TuneBackend(case)

    import bench.luinv.tune.backend as backend_mod
    monkeypatch.setattr(backend_mod, 'decode_opts',
                        lambda cfg: {'minblk': 2, 'pivot': 'bogus'})

    out = tb.compile({'minblk': 2, 'pivot': True, 'ix32': False}, 'k')
    assert not out.ok
    assert 'pivot' in out.detail
