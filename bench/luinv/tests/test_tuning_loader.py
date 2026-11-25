"""CPU-only tests of the tuning-search integration.

They prove the framework loads standalone from the aiter checkout (no torch, no
aiter package install, no GPU) and that the engine runs over our knob space.
Skipped when the framework is not checked out.
"""
import pytest

from bench.luinv.tune import loader
from bench.luinv.tune.space import BASELINE_CONFIG, decode_opts, make_space


@pytest.fixture(scope='module')
def ts():
    if not (loader.package_dir() / '__init__.py').is_file():
        pytest.skip(f'tuning framework not checked out at {loader.package_dir()}')
    return loader.load_tuning_search()


def test_loads_as_standalone_package(ts):
    # Registered under its own name, not under aiter (which would need torch)
    import sys

    assert 'tuning_search' in sys.modules
    assert ts.__name__ == 'tuning_search'
    assert hasattr(ts, 'tune') and hasattr(ts, 'CallableBackend')


def test_space_round_trips_baseline(ts):
    space = make_space(ts)
    genome = space.encode(BASELINE_CONFIG)
    config = space.decode(genome)

    assert config['minblk'] == BASELINE_CONFIG['minblk']
    assert config['pivot'] is False and config['ix32'] is False
    assert decode_opts(config) == {'minblk': 2}


def test_decode_opts_sets_transforms(ts):
    opts = decode_opts({'minblk': 1, 'pivot': True, 'ix32': True})
    assert opts == {'minblk': 1, 'pivot': 'shfl', 'ix32': 'on'}


def test_exhaustive_finds_synthetic_optimum(ts):
    # The engine over a deterministic synthetic objective must find the
    # enumerated optimum: exercises load -> SearchSpace -> tune end to end.
    synthetic = loader.load_submodule('synthetic')
    space, objective = synthetic.sphere(levels=5, dims=3)
    best_key, _, _ = synthetic.brute_force_optimum(space, objective)

    res = ts.tune(space, ts.SyntheticBackend(objective),
                  strategy='exhaustive', seed=0)

    assert res.best_key == best_key


def test_callable_backend_over_knob_space(ts):
    # Drive a CallableBackend over our actual space with a stand-in timer,
    # so the harness-side wiring (compile/check/measure signatures, baseline
    # confirmation) is covered without a GPU.
    space = make_space(ts)

    def compile_(config, key):
        return ts.CompileOutcome(True, artifact=config)

    def check(config, key, artifact):
        return True, 'ok'

    def measure(config, key, artifact, fidelity):
        # Prefer min-blocks 1; reward pivot; penalise ix32 a touch
        t = 1.0 + 0.1*int(config['minblk']) - 0.05*bool(config['pivot'])
        return [t]*fidelity.reps

    backend = ts.CallableBackend(compile_, check, measure)
    res = ts.tune(space, backend, strategy='exhaustive',
                  baseline=space.encode(BASELINE_CONFIG), seed=0)

    assert res.best_config['minblk'] == 1
    assert res.best_config['pivot'] is True
    # Confirmation against the baseline genome produced a speedup
    assert res.confirmed.get('speedup', 0) > 1.0
