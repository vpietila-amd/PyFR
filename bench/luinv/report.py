import json
from pathlib import Path

import pandas as pd


# Vendor peak rates; FLOP/s in TFLOP/s and bandwidth in TB/s
PEAKS = {
    'AMD Instinct MI300X': {'fp64': 81.7, 'fp64_mfma': 163.4,
                            'fp32': 163.4, 'hbm': 5.3},
    'AMD Instinct MI325X': {'fp64': 81.7, 'fp64_mfma': 163.4,
                            'fp32': 163.4, 'hbm': 6.0},
}

CASE_KEYS = ['n', 'precision', 'neles', 'ncolours', 'kind']


def load(paths):
    recs = []
    for p in paths:
        with Path(p).open() as f:
            recs.extend(json.loads(l) for l in f if l.strip())

    return recs


def bench_frame(recs):
    rows = []
    for r in recs:
        if r.get('mode') != 'bench':
            continue

        c, k = r['case'], (r['kernels'] or [{}])[0]
        rows.append({
            **{key: c[key] for key in CASE_KEYS},
            'variant': c['variant'],
            'timestamp': r['env']['timestamp'],
            'gpu': r['env'].get('gpu'),
            'ms': 1e3*r['rebuild']['median_s'],
            'tflops': r['rebuild']['nominal_tflops'],
            'gbs': r['rebuild']['compulsory_gbs'],
            'passed': (r['check'] or {}).get('passed'),
            'vgprs': k.get('vgprs'),
            'scratch': k.get('scratch_bytes'),
            'lds': k.get('lds_bytes'),
            'blocks_per_cu': k.get('blocks_per_cu'),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep the latest result of each case and variant
    df = df.sort_values('timestamp')
    return df.drop_duplicates(CASE_KEYS + ['variant'], keep='last')


def ref_frame(recs):
    rows = [{'ref': r['ref'], 'n': r['n'], 'neles': r['neles'],
             'dtype': r['dtype'], 'ms': 1e3*r['median_s'],
             'tflops': r['tflops']}
            for r in recs if r.get('mode') == 'ref' and 'n' in r]

    return pd.DataFrame(rows).drop_duplicates(['ref', 'n', 'neles', 'dtype'],
                                              keep='last')


def compare(recs, ref='baseline'):
    df = bench_frame(recs)
    if df.empty:
        return 'No benchmark records'

    out = []

    # Time of each variant and its speedup over the reference variant
    t = df.pivot_table(index=CASE_KEYS, columns='variant', values='ms')
    if ref in t:
        for v in t.columns:
            if v != ref:
                t[f'{v} speedup'] = t[ref] / t[v]

    out.append('Rebuild time [ms] (median)\n' + t.round(3).to_string())

    # Fraction of the peak rates of the GPU
    peaks = PEAKS.get(df['gpu'].iloc[-1])
    if peaks:
        df['% fp64 peak'] = 100*df['tflops'] / peaks['fp64']

    cols = ['tflops', 'gbs', '% fp64 peak', 'vgprs', 'scratch', 'lds',
            'blocks_per_cu', 'passed']
    cols = [c for c in cols if c in df]
    res = df.set_index(CASE_KEYS + ['variant'])[cols].sort_index()
    out.append('Rates and resources\n' + res.round(2).to_string())

    refs = ref_frame(recs)
    if not refs.empty:
        out.append('References\n' + refs.round(3).to_string(index=False))

    return '\n\n'.join(out)
