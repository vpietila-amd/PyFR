import numpy as np
import pytest

from bench.luinv.check import inverse_errors, summarise
from bench.luinv.inputs import KINDS, make_jacobians
from bench.luinv.problem import from_float64, to_float64


def _exact(kind, n=64, k=4, scale=0.05):
    a = np.eye(n) - scale*make_jacobians(kind, k, n, scale)
    return a, np.linalg.inv(a)


@pytest.mark.parametrize('kind', KINDS)
def test_exact_inverse_passes(kind):
    a, x = _exact(kind)
    assert summarise(inverse_errors(a, x, 'double'))['passed']


@pytest.mark.parametrize('prec', ['single', 'half', 'bf16'])
def test_rounded_inverse_passes(prec):
    from bench.luinv.problem import PRECISIONS

    # Rounding an exact inverse to the stored precision is acceptable
    a, x = _exact('jacobian')
    x = to_float64(from_float64(x, PRECISIONS[prec][0]))
    assert summarise(inverse_errors(a, x, prec))['passed']


def test_subnormal_rounding_passes():
    from bench.luinv.problem import PRECISIONS

    # Inverses with entries below the smallest normal half are still exact
    # up to the rounding which storing them implies
    a, x = _exact('nopivot', n=320)
    assert np.median(np.abs(x)) < np.finfo(np.float16).tiny
    x = to_float64(from_float64(x, PRECISIONS['half'][0]))
    assert summarise(inverse_errors(a, x, 'half'))['passed']


@pytest.mark.parametrize('rel', [1e-9, 1e-3])
def test_perturbed_inverse_fails(rel):
    # A single wrong entry well above roundoff must be caught
    a, x = _exact('jacobian', n=320)
    x[2, 5, 7] += rel*np.abs(x[2]).max()
    assert not summarise(inverse_errors(a, x, 'double'))['passed']


def test_swapped_inverses_fail():
    # Results written to the wrong element must be caught
    a, x = _exact('jacobian')
    assert not summarise(inverse_errors(a, x[[1, 0, 2, 3]], 'double'))[
        'passed'
    ]


def test_nonfinite_fails():
    a, x = _exact('jacobian')
    x[0, 0, 0] = np.nan
    assert not summarise(inverse_errors(a, x, 'double'))['passed']
