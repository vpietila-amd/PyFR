from contextlib import contextmanager
import math
from pathlib import Path

from pyfr.util import digest

from bench.luinv.rawhip import RawHIPModule, standalone


# Sources of the kernels which PyFR generates
SRC_DIR = Path(__file__).parent / 'build' / 'src'


@contextmanager
def record_kernels(backend):
    """Record the resource usage of the HIP kernels instantiated within."""
    kernels = []
    if backend.name != 'hip':
        yield kernels
        return

    prov = backend.pointwise
    names = {}

    def build_kernel(name, src, argtypes, *args, **kwargs):
        fun = build(name, src, argtypes, *args, **kwargs)
        names[id(fun)] = name, save_source(name, src)
        return fun

    def instantiate_kernel(dims, fun, args, meta):
        name, path = names.get(id(fun), (None, None))
        kernels.append(describe(fun, meta['block'], name, dims=list(dims),
                                compiler='hiprtc', source=path))
        return instantiate(dims, fun, args, meta)

    # Hand written HIP kernels report themselves when made
    def raw_kernel(self, name, argtypes, grid, block, sharedb=0):
        kern = raw(self, name, argtypes, grid, block, sharedb)
        kernels.append(describe(
            kern.fun, block, name, dims=list(grid), compiler=self.compiler,
            artifacts=str(self.artifacts) if self.artifacts else None,
            hipcc=self.remarks().get(name)
        ))
        return kern

    build, instantiate = prov._build_kernel, prov._instantiate_kernel
    raw = RawHIPModule.kernel

    prov._build_kernel = build_kernel
    prov._instantiate_kernel = instantiate_kernel
    RawHIPModule.kernel = raw_kernel
    try:
        yield kernels
    finally:
        del prov._build_kernel, prov._instantiate_kernel
        RawHIPModule.kernel = raw


def save_source(name, src):
    # Keep the exact source given to HIPRTC as a standalone .hip file
    path = SRC_DIR / f'{name}-{digest(src)[:16]}' / f'{name}.hip'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(standalone(src))

    return str(path)


def describe(fun, block, name=None, **kwargs):
    nthreads = math.prod(block)

    return {
        'name': name,
        'block': list(block),
        'vgprs': fun.nreg,
        'lds_bytes': fun.shared_mem,
        'scratch_bytes': fun.local_mem,
        'blocks_per_cu': fun.max_active_blocks(nthreads),
        **kwargs,
    }
