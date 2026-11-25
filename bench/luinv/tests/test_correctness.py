import itertools as it

import numpy as np
import pytest

from bench.luinv.check import inverse_errors, summarise
from bench.luinv.harness import Case, build, stage_and_invert
from bench.luinv.inputs import KINDS
from bench.luinv.problem import to_float64


def _invert_all(case, nsample=None):
    prob = build(case)
    refs = stage_and_invert(prob, case, nsample=nsample,
                            rng=np.random.default_rng(case.seed))
    return prob, refs


def _verify(prob, refs, precision, x=None):
    x = prob.read() if x is None else x
    eidx = np.concatenate([e for e, _ in refs.values()])
    a = np.concatenate([a for _, a in refs.values()])

    res = summarise(inverse_errors(a, x[eidx], precision))
    assert res['passed'], res


QUICK = [
    *[(n, p, 'jacobian', 65, 2) for n in (100, 320)
      for p in ('double', 'single', 'half', 'bf16')],
    (175, 'double', 'pivot', 130, 1),
    (175, 'double', 'dense', 64, 1),
    (320, 'double', 'jacobian', 513, 1),
]

FULL = [
    *[(n, p, k, 65, 1) for n, p, k in it.product(
        (100, 175, 320, 625), ('double', 'single', 'half', 'bf16'), KINDS
    )],
    # The other regimes of the size policy of hip_v2
    *[(n, p, k, 65, 1) for n, p, k in it.product(
        (70, 250, 400, 500), ('double', 'single'), KINDS
    )],
    *[(175, 'double', 'jacobian', e, 1)
      for e in (1, 7, 63, 64, 65, 512, 513, 1500)],
    (320, 'double', 'jacobian', 4096, 8),
]


def _ids(p):
    return 'n{}-{}-{}-e{}x{}'.format(*p)


@pytest.mark.parametrize('params', QUICK, ids=map(_ids, QUICK))
def test_inverse(variant, backend_name, params):
    n, prec, kind, neles, ncol = params
    case = Case(n=n, neles=neles, ncolours=ncol, precision=prec, kind=kind,
                variant=variant, backend=backend_name)

    _verify(*_invert_all(case, nsample=32), prec)


@pytest.mark.full
@pytest.mark.parametrize('params', FULL, ids=map(_ids, FULL))
def test_inverse_full(variant, backend_name, params):
    n, prec, kind, neles, ncol = params
    case = Case(n=n, neles=neles, ncolours=ncol, precision=prec, kind=kind,
                variant=variant, backend=backend_name)

    _verify(*_invert_all(case, nsample=32), prec)


def test_colour_isolation(variant, backend_name):
    case = Case(n=175, neles=200, ncolours=3, variant=variant,
                backend=backend_name)
    prob = build(case)
    colours = prob.colours

    # Poison every stored inverse then invert a single colour
    prob.fill(np.nan)
    refs = stage_and_invert(prob, case, colours=[1])
    x = prob.read()

    # Only the elements of that colour may have been written
    others = np.setdiff1d(np.arange(prob.neles), colours[1])
    assert np.isnan(x[others]).all()
    _verify(prob, refs, case.precision, x)

    # Inverting the remaining colours must cover every element
    refs |= stage_and_invert(prob, case, colours=[0, 2])
    x = prob.read()
    assert np.isfinite(x).all()
    _verify(prob, refs, case.precision, x)


def test_deterministic(variant, backend_name):
    case = Case(n=320, neles=300, variant=variant, backend=backend_name)
    prob, _ = _invert_all(case)
    x1 = prob.read()

    prob.fill(0)
    prob.invert(0, case.scale)
    x2 = prob.read()

    assert np.array_equal(x1, x2)


def test_rescale(variant, backend_name):
    # Inverting the staged Jacobians again with another scale must not
    # reuse any state from the first inversion
    case = Case(n=320, neles=100, variant=variant, backend=backend_name)
    prob, _ = _invert_all(case)

    scale = 2*case.scale
    prob.invert(0, scale)

    eidx = prob.colours[0]
    jac = to_float64(prob.jstage.get())[..., :len(eidx)].transpose(2, 0, 1)
    a = np.eye(case.n) - scale*jac

    res = summarise(inverse_errors(a, prob.read()[eidx], case.precision))
    assert res['passed'], res


@pytest.mark.parametrize('n', [100, 320])
def test_matches_baseline(variant, backend_name, n):
    if variant == 'baseline':
        pytest.skip('reference variant')

    xs = []
    for v in ('baseline', variant):
        case = Case(n=n, neles=130, ncolours=2, variant=v,
                    backend=backend_name)
        prob, refs = _invert_all(case)
        xs.append(prob.read())

    # Agreement to within the error bounds of the inverses themselves
    a = np.concatenate([a for _, a in refs.values()])
    eidx = np.concatenate([e for e, _ in refs.values()])
    xb, xv = xs[0][eidx], xs[1][eidx]

    err = inverse_errors(a, xb, case.precision)
    tol = err['kappa']*n*np.finfo(np.float64).eps
    rel = np.abs(xv - xb).sum(axis=-1).max(axis=-1)
    rel /= np.abs(xb).sum(axis=-1).max(axis=-1)
    assert (rel <= tol).all()
