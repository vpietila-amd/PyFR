import argparse
import json
from pathlib import Path
import sys

from bench.luinv import harness
from bench.luinv.inputs import KINDS
from bench.luinv.problem import PRECISIONS
from bench.luinv.variants import names


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


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m bench.luinv')
    sub = ap.add_subparsers(dest='cmd', required=True)

    _add_case_commands(sub)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
