import numpy as np

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.nputil import bf16_to_f32, bfloat16, f32_to_bf16, is_bf16


# Stored inverse and staged Jacobian dtypes of each precond-precision, as
# chosen by BlockJacobiPreconditioner
PRECISIONS = {
    'double': (np.float64, np.float64),
    'single': (np.float32, np.float32),
    'half': (np.float16, np.float32),
    'bf16': (bfloat16, bfloat16),
}

_backends = {}


def make_backend(name='hip', precision='double', memory_model='normal'):
    # Backends hold device state so build each kind only once per process
    key = (name, precision, memory_model)
    try:
        return _backends[key]
    except KeyError:
        pass

    cfg = Inifile()
    cfg.set('backend', 'precision', precision)
    cfg.set('backend', 'memory-model', memory_model)
    if name == 'hip':
        cfg.set('backend-hip', 'device-id', '0')

    backend = _backends[key] = get_backend(name, cfg)
    return backend


def to_float64(ary):
    if is_bf16(ary.dtype):
        return bf16_to_f32(ary).reshape(ary.shape).astype(np.float64)
    else:
        return ary.astype(np.float64)


def from_float64(ary, dtype):
    if is_bf16(dtype):
        return f32_to_bf16(ary).reshape(ary.shape).view(dtype)
    else:
        return ary.astype(dtype)


def read_tiled(mat):
    backend = mat.backend
    nmats, ntr, ntc = mat.nmats, mat.ntiles_r, mat.ntiles_c

    # Copy out the tiles of every matrix in its padded row-major order
    if backend.name == 'hip':
        backend.wait()

        buf = np.empty(mat.nbytes // mat.itemsize, dtype=mat.dtype)
        backend.hip.memcpy(buf, mat.data, mat.nbytes)

        buf = buf.reshape(nmats, ntr, ntc, mat.trows, mat.tcols)
        buf = buf.transpose(0, 1, 3, 2, 4)
    elif backend.name == 'openmp':
        buf = mat.data.transpose(0, 5, 1, 3, 2, 4)
    else:
        raise NotImplementedError(f'No tiled readback for {backend.name}')

    buf = buf.reshape(-1, mat.padr, mat.padc)[:nmats, :mat.nrow, :mat.ncol]
    return to_float64(np.ascontiguousarray(buf).view(mat.dtype))


class Problem:
    """The state the preconditioner keeps for one element type."""

    def __init__(self, backend, n, colours, variant, precision='double',
                 extent='luinv'):
        self.backend = backend
        self.n = n
        self.precision = precision
        self.pcdtype, self.wdtype = PRECISIONS[precision]

        # Element indices of each colour
        self.colours = {c: np.asarray(ix) for c, ix in colours.items()}
        self.neles = 1 + max(ix.max() for ix in self.colours.values())
        self.nemax = max(len(ix) for ix in self.colours.values())

        # Staged Jacobians of one colour and the inverses of all elements
        self.jstage = backend.matrix((n, n, self.nemax), dtype=self.wdtype,
                                     extent=extent, tags={'align'})
        self.inv = backend.tiled_matrix(block_shape=(n, n), nmats=self.neles,
                                        dtype=self.pcdtype)

        self.impl = variant(backend, self.jstage, self.inv, self.colours,
                            extent)
        backend.commit()

    def stage(self, colour, jac):
        # Lay the Jacobians of the colour out as the extract kernel does
        k = len(self.colours[colour])
        if jac.shape != (k, self.n, self.n):
            raise ValueError('Invalid Jacobian shape')

        ary = np.zeros((self.n, self.n, self.nemax))
        ary[..., :k] = jac.transpose(1, 2, 0)

        ary = from_float64(ary, self.wdtype)
        self.jstage.set(ary)

        # Return the Jacobians as staged, after any rounding
        return to_float64(ary[..., :k]).transpose(2, 0, 1)

    def invert(self, colour, scale):
        self.impl.invert(colour, scale)

    def wait(self):
        self.backend.wait()

    def read(self):
        return read_tiled(self.inv)

    def fill(self, value):
        # Overwrite every stored inverse
        ary = np.full((self.neles, self.n, self.n), value)
        self.inv._set(from_float64(ary, self.pcdtype))
        self.wait()


def make_colours(neles, ncolours, seed=0):
    # Scatter the elements over the colours as a mesh colouring would
    rng = np.random.default_rng(seed)
    cols = rng.permutation(np.arange(neles) % ncolours)

    return {c: np.flatnonzero(cols == c) for c in range(ncolours)}
