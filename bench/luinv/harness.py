from dataclasses import asdict, dataclass, field

import numpy as np

from bench.luinv.check import inverse_errors, summarise
from bench.luinv.env import environment
from bench.luinv.inputs import make_jacobians
from bench.luinv.problem import Problem, make_backend, make_colours
from bench.luinv.resources import record_kernels
from bench.luinv.timing import Timer, stats
from bench.luinv.variants import get_variant


@dataclass
class Case:
    n: int
    neles: int
    ncolours: int = 1
    precision: str = 'double'
    kind: str = 'jacobian'
    scale: float = 0.05
    seed: int = 0
    variant: str = 'baseline'
    backend: str = 'hip'
    fpprec: str = 'double'
    memory_model: str = 'normal'
    extra: dict = field(default_factory=dict)

    def label(self):
        return (f'{self.variant}:n{self.n}:{self.precision}:e{self.neles}'
                f'x{self.ncolours}:{self.kind}')


def nominal_flops(n):
    # LU (2/3 n^3) + forward (1/3 n^3) + backward (n^3) solves for I
    return 2*n**3


def build(case):
    backend = make_backend(case.backend, case.fpprec, case.memory_model)
    colours = make_colours(case.neles, case.ncolours, case.seed)

    return Problem(backend, case.n, colours, get_variant(case.variant),
                   case.precision)


def stage_and_invert(prob, case, colours=None, nsample=None, rng=None):
    """Invert colours as the preconditioner does, returning the matrices of
    a sample of elements for checking."""
    refs = {}
    for c in (colours if colours is not None else prob.colours):
        eidx = prob.colours[c]
        jac = make_jacobians(case.kind, len(eidx), case.n, case.scale,
                             seed=(case.seed, c))
        jac = prob.stage(c, jac)
        prob.invert(c, case.scale)
        prob.wait()

        # Always check the first and last element of each colour
        if nsample and nsample < len(eidx):
            pick = rng.choice(np.arange(1, len(eidx) - 1), nsample - 2,
                              replace=False)
            pick = np.sort(np.r_[0, pick, len(eidx) - 1])
        else:
            pick = np.arange(len(eidx))

        a = np.eye(case.n) - case.scale*jac[pick]
        refs[c] = (eidx[pick], a)

    return refs


def run_check(case, nsample=64):
    prob = build(case)
    rng = np.random.default_rng(case.seed)

    with record_kernels(prob.backend) as kernels:
        refs = stage_and_invert(prob, case, nsample=nsample, rng=rng)

    x = prob.read()
    eidx = np.concatenate([e for e, _ in refs.values()])
    a = np.concatenate([a for _, a in refs.values()])
    errs = inverse_errors(a, x[eidx], case.precision)

    return {'mode': 'check', 'case': asdict(case),
            'env': environment(prob.backend), 'kernels': kernels,
            'check': summarise(errs)}


def run_bench(case, reps=20, warmup=2, verify=True):
    prob = build(case)
    timer = Timer(prob.backend)

    # Build the kernels with a first full pass, checking a few results
    with record_kernels(prob.backend) as kernels:
        refs = stage_and_invert(prob, case, nsample=8,
                                rng=np.random.default_rng(case.seed))

    if verify:
        x = prob.read()
        eidx = np.concatenate([e for e, _ in refs.values()])
        a = np.concatenate([a for _, a in refs.values()])
        check = summarise(inverse_errors(a, x[eidx], case.precision))
    else:
        check = None

    # Time the inversion of each colour from its staged Jacobians
    colours = {}
    for c, eidx in prob.colours.items():
        # With one colour its Jacobians are still staged
        if len(prob.colours) > 1:
            jac = make_jacobians(case.kind, len(eidx), case.n, case.scale,
                                 seed=(case.seed, c))
            prob.stage(c, jac)

        times = timer(lambda: prob.invert(c, case.scale), reps, warmup)
        colours[c] = {'neles': len(eidx), **stats(times)}

    # Summarise a full rebuild as the sum over the colours
    t = sum(v['median_s'] for v in colours.values())
    flops = nominal_flops(case.n)*case.neles
    nbytes = case.neles*case.n**2*(np.dtype(prob.wdtype).itemsize +
                                   np.dtype(prob.pcdtype).itemsize)

    return {
        'mode': 'bench', 'case': asdict(case),
        'env': environment(prob.backend), 'kernels': kernels,
        'check': check, 'colours': colours,
        'rebuild': {
            'median_s': t,
            'matrices_per_s': case.neles / t,
            'nominal_tflops': flops / t / 1e12,
            'compulsory_gbs': nbytes / t / 1e9,
        },
    }
