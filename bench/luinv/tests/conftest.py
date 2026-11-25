import gc

import pytest

from bench.luinv.variants import names


def pytest_addoption(parser):
    parser.addoption('--variant', default='baseline',
                     help='comma separated variants to test, or all')
    parser.addoption('--backend', default='hip')


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
