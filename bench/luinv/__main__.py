import argparse
import json
from pathlib import Path
import sys

from bench.luinv import harness
from bench.luinv.inputs import KINDS
from bench.luinv.problem import PRECISIONS
from bench.luinv.variants import names


RESULTS = Path(__file__).parent / 'results'


def _case_args(ap):
    ap.add_argument('--variant', default='baseline', choices=names())
    ap.add_argument('--n', type=int, default=320)
    ap.add_argument('--precision', default='double', choices=PRECISIONS,
                    help='precond-precision of the case')
    ap.add_argument('--neles', type=int, default=512)
    ap.add_argument('--ncolours', type=int, default=1)
    ap.add_argument('--kind', default='jacobian', choices=KINDS)
    ap.add_argument('--scale', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--backend', default='hip')
    ap.add_argument('--fpprec', default='double',
                    help='backend precision')
    ap.add_argument('--memory-model', default='normal',
                    choices=['normal', 'large'])
    ap.add_argument('--out', type=Path,
                    help='append the result as a JSON line to this file')


def _case(args):
    return harness.Case(n=args.n, neles=args.neles, ncolours=args.ncolours,
                        precision=args.precision, kind=args.kind,
                        scale=args.scale, seed=args.seed,
                        variant=args.variant, backend=args.backend,
                        fpprec=args.fpprec, memory_model=args.memory_model)


def _emit(res, out):
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('a') as f:
            f.write(json.dumps(res) + '\n')


def _print_kernels(res):
    for k in res['kernels']:
        print(f"  kernel {k['name']}: block {k['block']} grid {k['dims']}  "
              f"vgprs {k['vgprs']}  lds {k['lds_bytes']} B  "
              f"scratch {k['scratch_bytes']} B  "
              f"blocks/CU {k['blocks_per_cu']}")


def _print_check(chk):
    if chk:
        print(f"  check: {'PASS' if chk['passed'] else 'FAIL'}  "
              f"({chk['nchecked']} matrices, max res {chk['max_res']:.2e} "
              f"[{chk['max_res_scaled']:.2g}], max ferr {chk['max_ferr']:.2e} "
              f"[{chk['max_ferr_scaled']:.2g}], max kappa "
              f"{chk['max_kappa']:.1e})")


# Checking and timing one case

def cmd_check(args):
    case = _case(args)
    res = harness.run_check(case, nsample=args.nsample)

    print(case.label())
    _print_kernels(res)
    _print_check(res['check'])
    _emit(res, args.out)

    return 0 if res['check']['passed'] else 1


def cmd_bench(args):
    case = _case(args)
    res = harness.run_bench(case, reps=args.reps, warmup=args.warmup,
                            verify=not args.no_verify)

    print(case.label())
    _print_kernels(res)
    _print_check(res['check'])
    for c, t in res['colours'].items():
        print(f"  colour {c}: {t['neles']} elements  median "
              f"{t['median_s']*1e3:.3f} ms  min {t['min_s']*1e3:.3f} ms  "
              f"p90 {t['p90_s']*1e3:.3f} ms")

    rb = res['rebuild']
    print(f"  rebuild {rb['median_s']*1e3:.3f} ms  "
          f"{rb['matrices_per_s']:.0f} matrices/s  "
          f"{rb['nominal_tflops']:.2f} TFLOP/s (2n^3)  "
          f"{rb['compulsory_gbs']:.1f} GB/s (compulsory)")
    _emit(res, args.out)

    return 0 if res['check'] is None or res['check']['passed'] else 1


def _add_case_commands(sub):
    ap = sub.add_parser('check', help='verify the inverses')
    _case_args(ap)
    ap.add_argument('--nsample', type=int, default=64,
                    help='matrices checked per colour')
    ap.set_defaults(func=cmd_check)

    ap = sub.add_parser('bench', help='time the inversion')
    _case_args(ap)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--no-verify', action='store_true')
    ap.set_defaults(func=cmd_bench)

    ap = sub.add_parser('profile', help='one timed inversion, for profilers')
    _case_args(ap)
    ap.set_defaults(func=cmd_bench, reps=1, warmup=0, no_verify=True)


# Sweeps, comparisons and library references

def _ints(s):
    return [int(v) for v in s.split(',')]


def _strs(s):
    return s.split(',')


def _refs(n, neles, precision, reps):
    import numpy as np

    from bench.luinv.problem import make_backend
    from bench.luinv.refs import References

    refs = References(make_backend('hip'))
    dtype = np.float64 if precision == 'double' else np.float32
    nbytes = neles*n*n*np.dtype(dtype).itemsize

    return [refs.rocsolver(n, neles, dtype, reps=reps),
            refs.gemm(n, neles, dtype, reps=reps),
            refs.copy(nbytes, reps=reps)]


def cmd_refs(args):
    for res in _refs(args.n, args.neles, args.precision, args.reps):
        if 'tflops' in res:
            print(f"{res['ref']}: n {res['n']} x {res['neles']} "
                  f"{res['dtype']}  median {res['median_s']*1e3:.3f} ms  "
                  f"{res['tflops']:.2f} TFLOP/s (2n^3)")
        else:
            print(f"{res['ref']}: {res['nbytes']/1e6:.0f} MB  median "
                  f"{res['median_s']*1e3:.3f} ms  {res['gbs']:.0f} GB/s")
        _emit(res, args.out)


def cmd_sweep(args):
    import itertools as it

    out = args.out or RESULTS / 'sweep.jsonl'
    grid = it.product(_ints(args.n), _strs(args.precision),
                      _ints(args.neles), _strs(args.variant))

    for n, prec, neles, variant in grid:
        case = harness.Case(n=n, neles=neles, ncolours=args.ncolours,
                            precision=prec, kind=args.kind,
                            variant=variant, backend=args.backend,
                            memory_model=args.memory_model)
        try:
            res = harness.run_bench(case, reps=args.reps)
        except (NotImplementedError, RuntimeError) as e:
            print(f'{case.label()}: skipped ({e})')
            continue

        rb, chk = res['rebuild'], res['check']
        print(f"{case.label()}: {rb['median_s']*1e3:.3f} ms  "
              f"{rb['nominal_tflops']:.2f} TFLOP/s  "
              f"check {'PASS' if chk['passed'] else 'FAIL'}")
        _emit(res, out)

    # Library references at each size, once per precision
    if args.refs:
        for n, prec, neles in it.product(_ints(args.n), _strs(args.precision),
                                         _ints(args.neles)):
            if prec in ('double', 'single'):
                for res in _refs(n, neles, prec, args.reps):
                    _emit(res, out)

    print(f'Results appended to {out}')


def cmd_compare(args):
    from bench.luinv.report import compare, load

    print(compare(load(args.files), ref=args.ref))


def _add_benchmark_commands(sub):
    ap = sub.add_parser('sweep', help='benchmark a grid of cases')
    ap.add_argument('--variant', default='baseline',
                    help='comma separated variants')
    ap.add_argument('--n', default='175,320,625')
    ap.add_argument('--precision', default='double,single')
    ap.add_argument('--neles', default='512,4096')
    ap.add_argument('--ncolours', type=int, default=1)
    ap.add_argument('--kind', default='jacobian', choices=KINDS)
    ap.add_argument('--backend', default='hip')
    ap.add_argument('--memory-model', default='normal',
                    choices=['normal', 'large'])
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--refs', action='store_true',
                    help='also time the library references')
    ap.add_argument('--out', type=Path)
    ap.set_defaults(func=cmd_sweep)

    ap = sub.add_parser('compare', help='tabulate benchmark results')
    ap.add_argument('files', nargs='+', type=Path)
    ap.add_argument('--ref', default='baseline',
                    help='variant to compute speedups against')
    ap.set_defaults(func=cmd_compare)

    ap = sub.add_parser('refs', help='time rocSOLVER, GEMM and copy '
                        'references')
    ap.add_argument('--n', type=int, default=320)
    ap.add_argument('--neles', type=int, default=512)
    ap.add_argument('--precision', default='double',
                    choices=['double', 'single'])
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', type=Path)
    ap.set_defaults(func=cmd_refs)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m bench.luinv')
    sub = ap.add_subparsers(dest='cmd', required=True)

    _add_case_commands(sub)
    _add_benchmark_commands(sub)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
