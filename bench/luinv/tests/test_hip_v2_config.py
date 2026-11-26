import pytest

from bench.luinv.harness import Case, build


# The waves per block config.hpp picks for the factorisation and the solves,
# which the launches take from the kernels: (n, precision) -> (K1W, WPB),
# either side of where the policy changes them
POLICY = [
    (70, 'double', (4, 2)),
    (96, 'double', (4, 2)),
    (97, 'double', (4, 4)),
    (416, 'double', (4, 4)),
    (417, 'double', (8, 4)),
    (70, 'single', (4, 2)),
    (512, 'single', (4, 4)),
    (513, 'single', (8, 4)),
    (320, 'half', (4, 4)),
    (625, 'bf16', (8, 4)),
]


def _waves(variant, backend_name, n, prec):
    if variant != 'hip_v2':
        pytest.skip('hip_v2 only')

    case = Case(n=n, neles=65, precision=prec, variant=variant,
                backend=backend_name)
    return build(case).impl.waves


@pytest.mark.parametrize('n,prec,waves', POLICY)
def test_policy(variant, backend_name, n, prec, waves):
    assert _waves(variant, backend_name, n, prec) == waves


def test_override(variant, backend_name, monkeypatch):
    monkeypatch.setenv('LUINV_K1_WAVES', '8')
    monkeypatch.setenv('LUINV_STRIPS', '2')
    monkeypatch.setenv('LUINV_WPB', '4')

    assert _waves(variant, backend_name, 70, 'double') == (8, 4)
