"""Load the search-driven tuning framework from a local aiter checkout.

The framework lives in the aiter tree at ``aiter/utility/tuning_search/`` on
the branch ``users/vpietila/tuning-search-framework``.  Importing it the normal
way (``import aiter.utility.tuning_search``) would run ``aiter/__init__.py``,
which imports torch and the rest of aiter; the framework itself needs only the
standard library and NumPy.  So we load the package directly from its directory
and register it as the top-level module ``tuning_search``, exactly as the KB
PCIe all-reduce harness does.  Its own relative imports (``from .api import``)
then resolve against that name.

The checkout is found relative to this repository: a sibling ``aiter`` beside
the PyFR checkout (``~/git/aiter`` for ``~/git/PyFR``).  Override the aiter root
with the ``LUINV_TUNING_AITER`` environment variable.
"""
import importlib.util
import os
from pathlib import Path
import sys


# Repo root is .../PyFR; the default aiter checkout is its sibling
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_AITER = _REPO_ROOT.parent / 'aiter'

# Where the package sits inside an aiter tree
_PKG_RELPATH = Path('aiter') / 'utility' / 'tuning_search'

_MODULE_NAME = 'tuning_search'
_cached = None


def aiter_root():
    """The aiter checkout to load the framework from."""
    env = os.environ.get('LUINV_TUNING_AITER')
    return Path(env).expanduser() if env else _DEFAULT_AITER


def package_dir():
    return aiter_root() / _PKG_RELPATH


def load_tuning_search():
    """Import and return the ``tuning_search`` package, cached per process."""
    global _cached
    if _cached is not None:
        return _cached

    if _MODULE_NAME in sys.modules:
        _cached = sys.modules[_MODULE_NAME]
        return _cached

    pkg = package_dir()
    init = pkg / '__init__.py'
    if not init.is_file():
        raise RuntimeError(
            f'Tuning framework not found at {pkg}.  Check out the branch that '
            f"carries it:\n    cd {aiter_root()} && git checkout "
            'users/vpietila/tuning-search-framework\n'
            'or point LUINV_TUNING_AITER at an aiter checkout that has '
            'aiter/utility/tuning_search/.'
        )

    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME, init, submodule_search_locations=[str(pkg)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise

    _cached = module
    return module


def load_submodule(name):
    """Import ``tuning_search.<name>`` (e.g. ``synthetic``) and return it."""
    import importlib

    load_tuning_search()
    return importlib.import_module(f'{_MODULE_NAME}.{name}')
