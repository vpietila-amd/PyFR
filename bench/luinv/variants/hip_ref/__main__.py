from types import SimpleNamespace

import numpy as np

from pyfr.nputil import is_bf16

from bench.luinv.problem import PRECISIONS, make_backend
from bench.luinv.variants.baseline import LUInverse
from bench.luinv.variants.hip_ref import generate


def main():
    import argparse

    ap = argparse.ArgumentParser(
        prog='python -m bench.luinv.variants.hip_ref',
        description='Render hip_ref headers, e.g. for offline hipcc builds'
    )
    ap.add_argument('n', type=int, nargs='+')
    ap.add_argument('--precision', default='double', choices=PRECISIONS)
    args = ap.parse_args()

    backend = make_backend('hip')
    pcdtype, wdtype = PRECISIONS[args.precision]

    # The template arguments as the baseline LUInverse derives them
    lu = LUInverse.__new__(LUInverse)
    lu.backend = backend
    lu.cdtype = np.dtype(np.float32 if is_bf16(wdtype) else wdtype)

    for n in args.n:
        m = SimpleNamespace(dtype=wdtype, ioshape=(n, n, lu.max_nwork))
        out = SimpleNamespace(nrow=n, dtype=pcdtype)
        print(generate(backend, lu._tplargs(m, out)))


main()
