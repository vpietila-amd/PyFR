import datetime
import gc
from pathlib import Path

import pytest

from bench.luinv.variants import names


def pytest_addoption(parser):
    parser.addoption('--variant', default='baseline',
                     help='comma separated variants to test, or all')
    parser.addoption('--backend', default='hip')
    parser.addoption('--bench-out', type=Path,
                     help='JSONL file to append benchmark results to')
    parser.addoption('--bench-against', type=Path,
                     help='JSONL results to check for regressions against')
    parser.addoption('--max-slowdown', type=float, default=0.03,
                     help='largest tolerated relative slowdown')


def pytest_generate_tests(metafunc):
    if 'variant' in metafunc.fixturenames:
        opt = metafunc.config.getoption('variant')
        variants = names() if opt == 'all' else opt.split(',')
        metafunc.parametrize('variant', variants)


@pytest.fixture
def backend_name(request):
    return request.config.getoption('backend')


@pytest.fixture(autouse=True)
def _release_device_memory():
    # Problems hold large workspaces so free them between tests
    yield
    gc.collect()


@pytest.fixture(scope='session')
def bench_out(request):
    out = request.config.getoption('bench_out')
    if not out:
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        out = Path(__file__).parents[1] / 'results' / f'pytest-{stamp}.jsonl'

    out.parent.mkdir(parents=True, exist_ok=True)
    return out
