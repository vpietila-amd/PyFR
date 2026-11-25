import json

import pytest

from bench.luinv.harness import Case, run_bench
from bench.luinv.report import CASE_KEYS, bench_frame, load


# Matrix sizes of hex p=2, 3 and 4 at realistic colour sizes
CASES = [(n, p, e) for n, e in [(175, 4096), (320, 4096), (625, 2048)]
         for p in ('double', 'single')]


def _ids(p):
    return 'n{}-{}-e{}'.format(*p)


@pytest.mark.bench
@pytest.mark.parametrize('params', CASES, ids=map(_ids, CASES))
def test_bench(request, variant, backend_name, bench_out, params):
    n, prec, neles = params
    case = Case(n=n, neles=neles, precision=prec, variant=variant,
                backend=backend_name)

    res = run_bench(case, reps=10)
    with bench_out.open('a') as f:
        f.write(json.dumps(res) + '\n')

    assert res['check']['passed'], res['check']

    # Compare against an earlier run of the same case and variant
    against = request.config.getoption('bench_against')
    if against:
        df = bench_frame(load([against]))
        key = {k: getattr(case, k) for k in CASE_KEYS}
        prev = df.loc[(df[list(key)] == list(key.values())).all(axis=1) &
                      (df['variant'] == variant)]
        if prev.empty:
            pytest.skip('no earlier result for this case')

        slow = request.config.getoption('max_slowdown')
        t, tprev = 1e3*res['rebuild']['median_s'], prev['ms'].iloc[-1]
        assert t <= (1 + slow)*tprev, f'{t:.3f} ms vs {tprev:.3f} ms'
