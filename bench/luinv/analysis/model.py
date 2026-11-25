"""Work per matrix in each phase of the blocked right-looking LU inverse."""

import numpy as np


def phase_work(tplargs, isz_s, isz_f, isz_o):
    """FLOPs and compulsory bytes per matrix of each phase.

    Uses the blocking of the template: panels of width B over the padded
    size NP, and the solves of LU X = I in column chunks of width sc whose
    forward sweeps skip the zero rows above each chunk.
    """
    n, NP, B, sc = (tplargs[k] for k in ('n', 'npad', 'blk', 'sc'))
    NPB = NP // B

    flops = dict.fromkeys(['eliminate', 'trsm', 'schur', 'fwd_update',
                           'fwd_diag', 'bwd_update', 'bwd_diag'], 0)
    for pb in range(0, NP, B):
        nt = NP - pb - B

        # Scale and eliminate the rows below each pivot within the panel
        for j in range(B):
            m = NP - pb - j - 1
            flops['eliminate'] += m + 2*m*(B - j - 1)

        # Unit lower triangular solve of the U panel, then the update
        flops['trsm'] += B*(B - 1)*nt
        flops['schur'] += 2*B*nt*nt

    for cc in range(0, NP, sc):
        ib0 = cc // B
        for ib in range(ib0, NPB):
            flops['fwd_update'] += 2*B*(ib - ib0)*B*sc
            flops['fwd_diag'] += B*(B - 1)*sc

        for ib in range(NPB):
            flops['bwd_update'] += 2*B*(NPB - 1 - ib)*B*sc
            flops['bwd_diag'] += B*B*sc

    nbytes = {
        'build': n*n*isz_s + NP*NP*isz_f,
        'writeout': n*n*isz_f + n*n*isz_o,
    }

    return flops, nbytes


def itemsizes(precision):
    from pyfr.nputil import is_bf16

    from bench.luinv.problem import PRECISIONS

    pcdtype, wdtype = PRECISIONS[precision]
    fdtype = np.float32 if is_bf16(wdtype) else wdtype

    return tuple(np.dtype(d).itemsize for d in (wdtype, fdtype, pcdtype))
