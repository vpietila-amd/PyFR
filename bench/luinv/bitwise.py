"""Compare the inverses of two source trees of a variant bit for bit.

    python -m bench.luinv.bitwise dump OUT.npz [--variant hip_v2]
    python -m bench.luinv.bitwise compare A.npz B.npz

dump inverts a fixed grid of cases (sizes, precisions, input kinds and
launch sizes) with the variant of the tree it runs from and saves every
inverse; run it from each tree, then compare.  The grid covers each regime
of the size policy of hip_v2, including launches of few matrices.
"""
import argparse
import itertools as it
import sys

import numpy as np


SIZES = (70, 100, 175, 250, 320, 400, 500, 625)
PRECISIONS = ('double', 'single', 'half', 'bf16')
KINDS = ('jacobian', 'pivot', 'illcond')
NELES = (65, 1500)


def cases():
    for n, p, k, e in it.product(SIZES, PRECISIONS, KINDS, NELES):
        # The large launches only for the plain Jacobians, to bound the time
        if e == NELES[-1] and k != 'jacobian':
            continue

        yield n, p, k, e


def key(n, p, k, e):
    return f'n{n}-{p}-{k}-e{e}'


def dump(out, variant):
    from bench.luinv.harness import Case, build, stage_and_invert

    res = {}
    for n, p, k, e in cases():
        case = Case(n=n, neles=e, precision=p, kind=k, variant=variant)
        prob = build(case)
        stage_and_invert(prob, case)
        x = prob.read()
        res[key(n, p, k, e)] = x.view(np.uint8).copy()
        print(key(n, p, k, e), x.dtype, x.shape, flush=True)

    np.savez(out, **res)


def compare(a, b):
    a, b = np.load(a), np.load(b)
    bad = 0
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            print(f'{k}: missing from one side')
            bad += 1
        elif not np.array_equal(a[k], b[k]):
            nd = np.count_nonzero(a[k] != b[k])
            print(f'{k}: {nd} of {a[k].size} bytes differ')
            bad += 1

    print(f'{len(a)} cases, {bad} differ')
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    d = sub.add_parser('dump')
    d.add_argument('out')
    d.add_argument('--variant', default='hip_v2')

    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')

    args = ap.parse_args(argv)
    if args.cmd == 'dump':
        return dump(args.out, args.variant)
    else:
        return compare(args.a, args.b)


if __name__ == '__main__':
    sys.exit(main())
