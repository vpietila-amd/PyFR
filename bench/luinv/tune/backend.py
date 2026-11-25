"""Bridge the lu_inv bench harness to the tuning-search framework.

One framework config is one ``hip_opt`` knob configuration.  The three backend
callables map onto the harness:

- ``compile``  build a :class:`~bench.luinv.problem.Problem` with the config's
  knobs baked in, forcing the kernels to compile and capturing their resources;
- ``check``    stage Jacobians, invert, and verify the inverses against NumPy
  with the precision-scaled error bounds (``check.summarise``);
- ``measure``  time ``invert`` on the largest colour with the backend's HIP
  events, returned in seconds.

The compiled :class:`Problem` is the opaque artifact, reused across check and
measure.  Register spills are *not* grounds for rejection here (the baseline
kernel already spills), so callers pass ``reject_spill=False`` to ``tune``.
"""
from pathlib import Path

import numpy as np

from pyfr.util import digest

from bench.luinv import harness
from bench.luinv.check import inverse_errors, summarise
from bench.luinv.inputs import make_jacobians
from bench.luinv.problem import Problem, make_backend
from bench.luinv.resources import record_kernels
from bench.luinv.timing import Timer
from bench.luinv.variants.hip_opt import variant_with_opts

from bench.luinv.tune.space import decode_opts


# Sources whose change should invalidate cached timings
_HIP_OPT_DIR = Path(__file__).resolve().parents[1] / 'variants' / 'hip_opt'
_BASELINE_DIR = Path(__file__).resolve().parents[1] / 'variants' / 'baseline'


def _source_digest():
    files = [_BASELINE_DIR / 'lu_inv_baseline.mako',
             _HIP_OPT_DIR / '__init__.py']
    return digest(*[f.read_bytes() for f in files])


class TuneBackend:
    """A tuning-search backend over the ``hip_opt`` variant for one case."""

    def __init__(self, case, nsample=32):
        if case.backend != 'hip':
            raise RuntimeError('Tuning needs the HIP backend')

        self.case = case
        self.nsample = nsample
        self.backend = make_backend(case.backend, case.fpprec,
                                    case.memory_model)

    # -- framework environment (cache/checkpoint fingerprint) -------------
    def environment(self):
        props = self.backend.props
        return {
            'kernel': 'lu_inv',
            'variant': 'hip_opt',
            'arch': props['gcn_arch_name'],
            'hip_runtime': self.backend.compiler.version,
            'precision': self.case.precision,
            'memory_model': self.case.memory_model,
            'sources': _source_digest(),
        }

    # -- the three backend callables --------------------------------------
    def compile(self, config, key):
        opts = decode_opts(config)
        try:
            with record_kernels(self.backend) as kernels:
                prob = Problem(self.backend, self.case.n, self._colours(),
                               variant_with_opts(opts), self.case.precision)
                # Force every kernel to build so failures surface here
                prob.impl.kerns
        except (AssertionError, RuntimeError, ValueError) as e:
            return self._compile_outcome(False, detail=str(e))

        resources = {
            'scratch_bytes': max((k['scratch_bytes'] for k in kernels),
                                 default=0),
            'vgprs': max((k['vgprs'] for k in kernels), default=0),
        }
        return self._compile_outcome(True, artifact=prob, resources=resources)

    def check(self, config, key, prob):
        rng = np.random.default_rng(self.case.seed)
        refs = harness.stage_and_invert(prob, self.case, nsample=self.nsample,
                                        rng=rng)
        x = prob.read()
        eidx = np.concatenate([e for e, _ in refs.values()])
        a = np.concatenate([a for _, a in refs.values()])
        s = summarise(inverse_errors(a, x[eidx], self.case.precision))

        detail = (f"res {s['max_res_scaled']:.2g}, ferr "
                  f"{s['max_ferr_scaled']:.2g}")
        return bool(s['passed']), detail

    def measure(self, config, key, prob, fidelity):
        c = max(prob.colours, key=lambda k: len(prob.colours[k]))
        eidx = prob.colours[c]
        jac = make_jacobians(self.case.kind, len(eidx), self.case.n,
                             self.case.scale, seed=(self.case.seed, c))
        prob.stage(c, jac)

        timer = Timer(self.backend)
        ms = timer(lambda: prob.invert(c, self.case.scale),
                   reps=fidelity.reps, warmup=fidelity.warmup)
        # Timer uses HIP event elapsed_time, which is in milliseconds
        return [float(t) / 1e3 for t in ms]

    # -- helpers ----------------------------------------------------------
    def _colours(self):
        from bench.luinv.problem import make_colours

        return make_colours(self.case.neles, self.case.ncolours,
                            self.case.seed)

    def _compile_outcome(self, ok, **kwargs):
        from bench.luinv.tune.loader import load_tuning_search

        return load_tuning_search().CompileOutcome(ok, **kwargs)


def make_backend_and_space(case, tuning_search, nsample=32):
    """Return ``(callable_backend, space, baseline_genome)`` for a case."""
    from bench.luinv.tune.space import BASELINE_CONFIG, make_space

    tb = TuneBackend(case, nsample=nsample)
    backend = tuning_search.CallableBackend(
        tb.compile, tb.check, tb.measure, environment=tb.environment()
    )
    space = make_space(tuning_search)
    baseline = space.encode(BASELINE_CONFIG)
    return backend, space, baseline
