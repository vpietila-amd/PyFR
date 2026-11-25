import numpy as np

from pyfr.nputil import is_bf16

from bench.luinv.problem import PRECISIONS


def roundoffs(precision):
    pcdtype, wdtype = PRECISIONS[precision]

    # Factorisation is in the staging precision, with bf16 done in single
    fdtype = np.float32 if is_bf16(wdtype) else wdtype
    uf = float(np.finfo(fdtype).eps) / 2

    # Relative and absolute (underflow) rounding errors of the stored inverse
    if is_bf16(pcdtype):
        uo, eta = 2.0**-8, 2.0**-134
    else:
        uo = float(np.finfo(pcdtype).eps) / 2
        eta = float(np.finfo(pcdtype).smallest_subnormal) / 2

    return uf, uo, eta


def _infnorm(m):
    return np.abs(m).sum(axis=-1).max(axis=-1)


def inverse_errors(a, x, precision):
    """Per matrix error metrics of the inverses x of a.

    The scaled metrics are relative to first order error bounds, so values
    of O(1) or less are acceptable whatever the precision:

      res  = ||a x - I|| / (||a|| ||x|| (n uf + uo + n eta / ||x||))
      ferr = ||x - inv(a)|| / (||inv(a)|| (kappa n uf + uo
                                           + n eta / ||inv(a)||))

    where uf and uo are the unit roundoffs of the factorisation and of the
    stored inverse, eta the absolute rounding error of its subnormals, and
    kappa the condition number of a; all norms are infinity norms.  The eta
    terms matter for half precision, where small entries of an inverse are
    subnormal.
    """
    n = a.shape[-1]
    uf, uo, eta = roundoffs(precision)

    xref = np.linalg.inv(a)
    na, nx, nxref = _infnorm(a), _infnorm(x), _infnorm(xref)
    kappa = na*nxref

    res = _infnorm(a @ x - np.eye(n)) / (na*nx)
    ferr = _infnorm(x - xref) / nxref

    return {
        'finite': np.isfinite(x).all(axis=(-2, -1)),
        'kappa': kappa,
        'res': res,
        'ferr': ferr,
        'res_scaled': res / (n*uf + uo + n*eta/nx),
        'ferr_scaled': ferr / (kappa*n*uf + uo + n*eta/nxref),
    }


def summarise(errs, tol=1.0):
    worst = {k: float(np.max(v)) for k, v in errs.items() if k != 'finite'}

    return {
        'nchecked': int(len(errs['finite'])),
        'all_finite': bool(errs['finite'].all()),
        **{f'max_{k}': v for k, v in worst.items()},
        'passed': bool(errs['finite'].all() and
                       worst['res_scaled'] <= tol and
                       worst['ferr_scaled'] <= tol),
    }
