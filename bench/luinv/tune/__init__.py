"""Search-driven tuning of the lu_inv ``hip_opt`` variant.

Loads the kernel-neutral tuning-search framework from a local aiter checkout
(:mod:`bench.luinv.tune.loader`) and drives it over the ``hip_opt`` knob space
(:mod:`bench.luinv.tune.space`) through a bench-harness backend
(:mod:`bench.luinv.tune.backend`).
"""
from bench.luinv.tune.backend import TuneBackend, make_backend_and_space
from bench.luinv.tune.loader import load_submodule, load_tuning_search
from bench.luinv.tune.space import BASELINE_CONFIG, decode_opts, make_space

__all__ = [
    'BASELINE_CONFIG',
    'TuneBackend',
    'decode_opts',
    'load_submodule',
    'load_tuning_search',
    'make_backend_and_space',
    'make_space',
]
