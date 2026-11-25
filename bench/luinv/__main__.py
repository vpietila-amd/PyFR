import argparse
from dataclasses import asdict
import json
import os
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


# Offline hipcc builds of hand written kernels

def cmd_hipcc(args):
    os.environ['LUINV_COMPILER'] = 'hipcc'
    res = harness.run_check(_case(args), nsample=4)

    cmds = {}
    for k in res['kernels']:
        if k.get('artifacts'):
            print(f"{k['name']}: {k['artifacts']}")
            for key, val in (k['hipcc'] or {}).items():
                print(f'  {key}: {val}')

            adir = Path(k['artifacts'])
            cmd = json.loads((adir / 'command.json').read_text())
            cmds[cmd[-3]] = cmd

    if not cmds:
        print('No hand written HIP kernels in this variant', file=sys.stderr)
        return 1

    # Let clangd and friends see the exact compile commands
    ccpath = Path(__file__).parent / 'build' / 'compile_commands.json'
    try:
        db = {e['file']: e for e in json.loads(ccpath.read_text())}
    except FileNotFoundError:
        db = {}

    for f, cmd in cmds.items():
        db[f] = {'directory': str(Path(f).parent), 'file': f,
                 'arguments': cmd}

    ccpath.write_text(json.dumps(list(db.values()), indent=1))
    print(f'Updated {ccpath}')

    return 0 if res['check']['passed'] else 1


def _add_hipcc_commands(sub):
    ap = sub.add_parser('hipcc', help='build hand written kernels with hipcc '
                        'and report their resources')
    _case_args(ap)
    ap.set_defaults(func=cmd_hipcc)


# Static analysis and profiling

def _case_argv(args):
    return ['--variant', args.variant, '--n', str(args.n),
            '--precision', args.precision, '--neles', str(args.neles),
            '--ncolours', str(args.ncolours), '--kind', args.kind,
            '--scale', str(args.scale), '--seed', str(args.seed),
            '--backend', args.backend, '--fpprec', args.fpprec,
            '--memory-model', args.memory_model]


def cmd_resources(args):
    from bench.luinv.analysis.isa import assembly_file, isa_stats
    from bench.luinv.problem import make_backend
    from bench.luinv.rawhip import RawHIPModule

    # Build every kernel of the variant offline with hipcc
    os.environ['LUINV_COMPILER'] = 'hipcc'
    case = _case(args)
    res = harness.run_check(case, nsample=4)
    backend = make_backend(case.backend, case.fpprec, case.memory_model)

    rows = []
    for k in res['kernels']:
        if k.get('source'):
            mod = RawHIPModule(backend, k['source'], compiler='hipcc')
            adir, hipcc = mod.artifacts, mod.remarks().get(k['name'], {})
        else:
            adir, hipcc = k['artifacts'], k['hipcc'] or {}

        rows.append({'kernel': k['name'], 'grid': k['dims'][0],
                     'block': k['block'][0], 'artifacts': str(adir),
                     **hipcc, **isa_stats(assembly_file(adir), k['name'])})

    for r in rows:
        print(f"{r['kernel']} (grid {r['grid']}, block {r['block']})")
        for key, val in r.items():
            if key not in ('kernel', 'grid', 'block'):
                print(f'  {key}: {val}')

    _emit({'mode': 'resources', 'case': res['case'], 'env': res['env'],
           'kernels': rows}, args.out)


def cmd_counters(args):
    from bench.luinv.analysis.profile import GROUPS, collect

    case = _case(args)
    groups = GROUPS if args.groups == 'all' else {
        g: GROUPS[g] for g in args.groups.split(',')
    }
    outdir = args.outdir or (RESULTS / 'profile' /
                             case.label().replace(':', '_'))

    kernels = collect(_case_argv(args), outdir, groups)

    nominal = harness.nominal_flops(case.n)
    for name, c in kernels.items():
        print(f"{name}: {c['duration_s']*1e3:.3f} ms per dispatch")
        if 'flops_f64' in c or 'flops_f32' in c:
            # Each dispatch inverts at most a launch's worth of matrices
            flops = c.get('flops_f64') or c.get('flops_f32')
            print(f'  measured FLOPs per dispatch {flops:.3e}')
        for key in ('tflops_f64', 'tflops_f32', 'fabric_gbs',
                    'intensity_fabric', 'l2_hit_pct', 'valu_active_pct',
                    'lds_active_pct', 'wait_inst_pct', 'wait_any_pct',
                    'lds_conflict_cycles_per_inst'):
            if c.get(key) is not None:
                print(f'  {key}: {c[key]:.3f}')

    _emit({'mode': 'counters', 'case': asdict(case),
           'nominal_flops_per_matrix': nominal, 'kernels': kernels,
           'outdir': str(outdir)}, args.out)


def cmd_roofline(args):
    from bench.luinv.analysis import roofline

    case = _case(args)
    workdir = args.outdir or (RESULTS / 'rocprof-compute' /
                              case.label().replace(':', '_'))

    print(f'Profiling into {workdir} (this replays the case many times)')
    roofline.profile(_case_argv(args), workdir, 'luinv', kernel=args.kernel,
                     roof_only=args.roof_only)
    out = roofline.analyze(workdir)
    print(f'Analysis in {out}; plots and raw data under {workdir}')


def _add_analysis_commands(sub):
    ap = sub.add_parser('resources', help='hipcc resource usage and ISA '
                        'statistics of the kernels of a variant')
    _case_args(ap)
    ap.set_defaults(func=cmd_resources)

    ap = sub.add_parser('counters', help='hardware counters through '
                        'rocprofv3, one pass per counter group')
    _case_args(ap)
    ap.add_argument('--groups', default='all',
                    help='comma separated counter groups, or all')
    ap.add_argument('--outdir', type=Path)
    ap.set_defaults(func=cmd_counters)

    ap = sub.add_parser('roofline', help='rocprof-compute profile with '
                        'empirical roofline, then analyze')
    _case_args(ap)
    ap.add_argument('--kernel', help='kernel name filter')
    ap.add_argument('--roof-only', action='store_true')
    ap.add_argument('--outdir', type=Path)
    ap.set_defaults(func=cmd_roofline)


# Phases of the baseline kernel

def cmd_phases(args):
    case = _case(args)
    res = harness.run_phases(case, reps=args.reps)

    over = res['instrumented_median_s'] / res['median_s'] - 1
    print(f"{case.label()}: {res['median_s']*1e3:.3f} ms per launch, "
          f"{res['instrumented_median_s']*1e3:.3f} ms instrumented "
          f"({100*over:+.1f}%)")
    print(f"  mean block lifetime {res['block_s']*1e6:.0f} us")
    for k, v in res['phases'].items():
        rate = (f"  {v['tflops']:6.2f} TFLOP/s" if 'tflops' in v else
                f"  {v['gbs']:6.0f} GB/s" if 'gbs' in v else '')
        print(f"  {k:12s} {v['block_s']*1e6:10.1f} us  "
              f"{100*v['share']:5.1f}%{rate}")

    _emit(res, args.out)


def _add_phase_commands(sub):
    ap = sub.add_parser('phases', help='time the phases of the baseline '
                        'kernel through instrumented hip_ref')
    _case_args(ap)
    ap.add_argument('--reps', type=int, default=10)
    ap.set_defaults(func=cmd_phases)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m bench.luinv')
    sub = ap.add_subparsers(dest='cmd', required=True)

    _add_case_commands(sub)
    _add_benchmark_commands(sub)
    _add_hipcc_commands(sub)
    _add_analysis_commands(sub)
    _add_phase_commands(sub)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
