import importlib

# Implementations with the interface of LUInverse, by name
_variants = {
    # The live implementation on this branch
    'pyfr': 'pyfr.integrators.implicit.luinv:LUInverse',

    # Frozen copy of the implementation before the changes on this branch
    'baseline': 'bench.luinv.variants.baseline:LUInverse',
}


def names():
    return list(_variants)


def get_variant(name):
    try:
        mod, cls = _variants[name].split(':')
    except KeyError:
        raise ValueError(f'Unknown variant {name!r}; must be one of '
                         f'{", ".join(_variants)}') from None

    return getattr(importlib.import_module(mod), cls)
