import pytest

from bench.luinv.harness import Case, build


# The knobs config.hpp picks: (n, precision, matrices) -> (K1W, S, K1MB,
# IOL, WPB, K2MB), as documented there; 65 matrices make a launch of few
POLICY = [
    (70, 'double', 65, (4, 1, 2, 0, 2, 2)),
    (70, 'double', 1500, (4, 1, 4, 1, 2, 2)),
    (175, 'double', 1500, (4, 1, 2, 1, 4, 2)),
    (320, 'double', 1500, (4, 1, 2, 0, 4, 2)),
    (400, 'double', 65, (4, 1, 2, 0, 4, 1)),
    (500, 'double', 65, (8, 2, 1, 0, 4, 1)),
    (70, 'single', 65, (4, 1, 4, 1, 2, 2)),
    (200, 'single', 65, (4, 1, 3, 1, 4, 2)),
    (320, 'half', 65, (4, 2, 3, 1, 4, 2)),
    (400, 'bf16', 65, (4, 1, 3, 1, 4, 1)),
    (500, 'single', 65, (4, 2, 2, 1, 4, 1)),
    (625, 'single', 65, (8, 2, 1, 0, 4, 1)),
]

KEYS = ('k1w', 'strips', 'k1minblk', 'iol', 'wpb', 'k2minblk')


def _opts(variant, backend_name, n, prec, neles):
    if variant != 'hip_v2':
        pytest.skip('hip_v2 only')

    case = Case(n=n, neles=neles, precision=prec, variant=variant,
                backend=backend_name)
    return build(case).impl.opts


@pytest.mark.parametrize('n,prec,neles,knobs', POLICY)
def test_policy(variant, backend_name, n, prec, neles, knobs):
    opts = _opts(variant, backend_name, n, prec, neles)

    assert opts['npad'] == -(-n // 16)*16
    assert tuple(opts[k] for k in KEYS) == knobs


def test_override(variant, backend_name, monkeypatch):
    monkeypatch.setenv('LUINV_K1_WAVES', '8')
    monkeypatch.setenv('LUINV_STRIPS', '2')
    monkeypatch.setenv('LUINV_WPB', '4')
    opts = _opts(variant, backend_name, 70, 'double', 65)

    assert (opts['k1w'], opts['strips'], opts['wpb']) == (8, 2, 4)
