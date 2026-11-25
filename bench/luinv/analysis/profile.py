"""Hardware counters of a case through rocprofv3, one pass per group."""

import csv
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import sys

from bench.luinv.env import ROOT


GROUPS = {
    'flops64': ['SQ_INSTS_VALU_ADD_F64', 'SQ_INSTS_VALU_MUL_F64',
                'SQ_INSTS_VALU_FMA_F64', 'SQ_INSTS_VALU_TRANS_F64',
                'SQ_INSTS_VALU_MFMA_MOPS_F64'],
    'flops32': ['SQ_INSTS_VALU_ADD_F32', 'SQ_INSTS_VALU_MUL_F32',
                'SQ_INSTS_VALU_FMA_F32', 'SQ_INSTS_VALU_TRANS_F32',
                'SQ_INSTS_VALU_MFMA_MOPS_F32'],
    # The derived sizes each take a pass of their own
    'fabric_read': ['FETCH_SIZE'],
    'fabric_write': ['WRITE_SIZE'],
    'l2': ['TCC_HIT_sum', 'TCC_MISS_sum', 'TCC_EA0_RDREQ_sum',
           'TCC_EA0_WRREQ_sum'],
    'l1': ['TCP_TCC_READ_REQ_sum', 'TCP_TCC_WRITE_REQ_sum',
           'TA_FLAT_READ_WAVEFRONTS_sum', 'TA_FLAT_WRITE_WAVEFRONTS_sum'],
    'insts': ['SQ_INSTS_VALU', 'SQ_INSTS_SALU', 'SQ_INSTS_LDS',
              'SQ_INSTS_VMEM_RD', 'SQ_INSTS_VMEM_WR', 'SQ_INSTS_SMEM',
              'SQ_INSTS_BRANCH'],
    'waves': ['SQ_WAVES', 'SQ_WAVE_CYCLES', 'SQ_BUSY_CYCLES',
              'GRBM_GUI_ACTIVE'],
    'stalls': ['SQ_WAIT_ANY', 'SQ_WAIT_INST_ANY', 'SQ_WAIT_INST_LDS',
               'SQ_ACTIVE_INST_VALU', 'SQ_ACTIVE_INST_LDS'],
    'lds': ['SQ_LDS_BANK_CONFLICT', 'SQ_INST_LEVEL_LDS'],
}


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run_pass(case_argv, counters, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = ['rocprofv3', '--kernel-trace', '--output-format', 'csv',
           '-d', str(outdir), '-o', 'run']
    if counters:
        cmd += ['--pmc', *counters]
    cmd += ['--', sys.executable, '-m', 'bench.luinv', 'profile',
            *case_argv]

    env = dict(os.environ, PYTHONPATH=str(ROOT))
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                         text=True)
    if res.returncode:
        raise RuntimeError(f'rocprofv3 failed:\n{res.stderr[-4000:]}')

    # Per kernel durations, ignoring the runtime's own copy kernels
    durs = defaultdict(list)
    for r in _read_csv(outdir / 'run_kernel_trace.csv'):
        if not r['Kernel_Name'].startswith('__amd'):
            dt = int(r['End_Timestamp']) - int(r['Start_Timestamp'])
            durs[r['Kernel_Name']].append(dt*1e-9)

    # Per kernel counter values, averaged over the dispatches
    vals = defaultdict(lambda: defaultdict(list))
    if counters:
        for r in _read_csv(outdir / 'run_counter_collection.csv'):
            if not r['Kernel_Name'].startswith('__amd'):
                vals[r['Kernel_Name']][r['Counter_Name']].append(
                    float(r['Counter_Value'])
                )

    return durs, {k: {c: sum(v) / len(v) for c, v in cs.items()}
                  for k, cs in vals.items()}


def collect(case_argv, outdir, groups=GROUPS):
    kernels = defaultdict(dict)
    durs = {}
    for g, counters in groups.items():
        d, vals = run_pass(case_argv, counters, Path(outdir) / g)
        for k, cs in vals.items():
            kernels[k].update(cs)

        for k, v in d.items():
            durs.setdefault(k, []).extend(v)

    for k, v in durs.items():
        v.sort()
        kernels[k]['duration_s'] = v[len(v) // 2]
        kernels[k]['dispatches_per_pass'] = len(v) // len(groups)

    return {k: derive(v) for k, v in kernels.items()}


def derive(c):
    g = c.get
    out = dict(c)
    t = c['duration_s']

    # FLOPs as rocprof-compute counts them, assuming full wavefronts
    if 'SQ_INSTS_VALU_FMA_F64' in c:
        f64 = 64*(g('SQ_INSTS_VALU_ADD_F64') + g('SQ_INSTS_VALU_MUL_F64') +
                  g('SQ_INSTS_VALU_TRANS_F64') +
                  2*g('SQ_INSTS_VALU_FMA_F64'))
        out['flops_f64'] = f64 + 512*g('SQ_INSTS_VALU_MFMA_MOPS_F64')
        out['tflops_f64'] = out['flops_f64'] / t / 1e12

    if 'SQ_INSTS_VALU_FMA_F32' in c:
        f32 = 64*(g('SQ_INSTS_VALU_ADD_F32') + g('SQ_INSTS_VALU_MUL_F32') +
                  g('SQ_INSTS_VALU_TRANS_F32') +
                  2*g('SQ_INSTS_VALU_FMA_F32'))
        out['flops_f32'] = f32 + 256*g('SQ_INSTS_VALU_MFMA_MOPS_F32')
        out['tflops_f32'] = out['flops_f32'] / t / 1e12

    # Traffic from L2 to the fabric (MALL and HBM); the sizes are in KiB
    if 'FETCH_SIZE' in c:
        out['fabric_bytes'] = 1024*(g('FETCH_SIZE') + g('WRITE_SIZE'))
        out['fabric_gbs'] = out['fabric_bytes'] / t / 1e9

    if 'TCC_HIT_sum' in c:
        acc = g('TCC_HIT_sum') + g('TCC_MISS_sum')
        out['l2_hit_pct'] = 100*g('TCC_HIT_sum') / acc if acc else None

    if 'SQ_WAVE_CYCLES' in c and 'SQ_WAIT_INST_ANY' in c:
        wc = g('SQ_WAVE_CYCLES')
        out['wait_inst_pct'] = 100*g('SQ_WAIT_INST_ANY') / wc
        out['wait_any_pct'] = 100*g('SQ_WAIT_ANY') / wc
        out['valu_active_pct'] = 100*g('SQ_ACTIVE_INST_VALU') / wc
        out['lds_active_pct'] = 100*g('SQ_ACTIVE_INST_LDS') / wc

    if 'SQ_INSTS_LDS' in c and 'SQ_LDS_BANK_CONFLICT' in c:
        out['lds_conflict_cycles_per_inst'] = (g('SQ_LDS_BANK_CONFLICT') /
                                              max(g('SQ_INSTS_LDS'), 1))

    flops = out.get('flops_f64') or out.get('flops_f32')
    if flops and 'fabric_bytes' in out:
        out['intensity_fabric'] = flops / out['fabric_bytes']

    return out
