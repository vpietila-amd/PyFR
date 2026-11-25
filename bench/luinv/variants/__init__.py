import importlib

# Implementations with the interface of LUInverse, by name
_variants = {
    # The live implementation on this branch
    'pyfr': 'pyfr.integrators.implicit.luinv:LUInverse',

    # Frozen copy of the implementation before the changes on this branch
    'baseline': 'bench.luinv.variants.baseline:LUInverse',

    # The baseline kernel compiled as hand written HIP
    'hip_ref': 'bench.luinv.variants.hip_ref:LUInverse',

    # The same with the time spent in each phase recorded
    'hip_ref_phases': 'bench.luinv.variants.hip_ref:PhasedLUInverse',
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
