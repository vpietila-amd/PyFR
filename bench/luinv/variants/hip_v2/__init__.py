"""The LU inverse in separate HIP kernels for the build, the factorisation
and the solves, hand written with MFMA (lu_inv.hip and include/luinv/hip_v2).

lu_build forms I - scale*src in a strip-major workspace; lu_factor_mfma
factorises it left-looking with a block of four or eight waves per matrix,
leaving the factor packed for the MFMA; lu_solve_mfma solves for 16 columns
of the inverse per wave.  The matrices are padded to a multiple of 16.  The
factorisation is in fp64 for double and in fp32 for the other precisions.

The kernels are C++ templates over their configuration (config.hpp): the
problem, which this passes as -D macros, and the tuning knobs, which the
policy there picks for the size and precision.  The launch bounds of the
factorisation and the solves carry their block sizes, which this reads back
to size their launches.  The environment can override the knobs:
  LUINV_K1_WAVES=<n>  waves per block of the factorisation, 4 or 8
  LUINV_STRIPS=<n>    strips of 16 columns per pass of the factorisation
  LUINV_K1_MINBLK=<n> launch bounds of the factorisation: the minimum waves
                      per SIMD
  LUINV_INV_OOL=0|1   invert the diagonal blocks out of line
  LUINV_WPB=<n>       waves per block of the solves, which share their loads
                      of the factor
  LUINV_K2_MINBLK=<n> launch bounds of the solves: the minimum waves per SIMD
and for experiments
  LUINV_NWORK=<n>     most matrices factorised at once
  LUINV_PHASES=1      count the cycles of the phases of the factorisation,
                      for read_ticks
  LUINV_DEFS=A=1,B    extra macros
"""
from functools import cached_property
import os
from pathlib import Path

import numpy as np

from pyfr.nputil import npdtype_to_ctype

from bench.luinv.rawhip import PTR, RawHIPModule
from bench.luinv.variants.baseline import LUInverse as BaselineLUInverse


HERE = Path(__file__).parent
INCLUDE = HERE / 'include'

BUILD, FACTOR = 'lu_build_hip_v2', 'lu_factor_mfma_hip_v2'
SOLVE = 'lu_solve_mfma_hip_v2'

# C types of the storage types
CTYPES = {'double': 'double', 'float': 'float', 'half': '_Float16',
          'bf16': 'luinv::bf16'}

# Macros of the knobs for their environment variables
KNOBS = {'LUINV_K1_WAVES': 'LUINV_K1W', 'LUINV_STRIPS': 'LUINV_S',
         'LUINV_K1_MINBLK': 'LUINV_K1MB', 'LUINV_INV_OOL': 'LUINV_IOL',
         'LUINV_WPB': 'LUINV_WPB', 'LUINV_K2_MINBLK': 'LUINV_K2MB'}


class LUInverse(BaselineLUInverse):
    def _tplargs(self, m, out):
        t = super()._tplargs(m, out)
        if t['ftype'] not in ('double', 'float'):
            raise ValueError(f'No hip_v2 factorisation in {t["ftype"]}')

        # Pad to whole strips of 16 columns; the workspace holds the packed
        # factor, then the inverses of the diagonal blocks of U, as
        # config.hpp has it
        t['npad'] = npad = -(-t['n'] // 16)*16
        t['wsz'] = npad*(npad + 16)

        # Factorise more matrices at once, within a 2 GiB workspace
        nwork = int(os.environ.get('LUINV_NWORK', 0))
        nwork = nwork or max(512, min(4096, 2**28 // t['wsz']))

        # Starting each part at a whole AoSoA group of the matrices
        t['nwork'] = min(nwork // 64 * 64, m.ioshape[2])

        return t

    @cached_property
    def ticks(self):
        # Cycles per phase of the factorisation, with LUINV_PHASES set
        return self.backend.hip.mem_alloc(8*16)

    def read_ticks(self):
        self.backend.wait()
        out = np.zeros(16, dtype=np.uint64)
        self.backend.hip.memcpy(out, self.ticks, out.nbytes)
        self.backend.hip.memset(self.ticks, 0, out.nbytes)
        return out

    @cached_property
    def module(self):
        b, t, out, env = self.backend, self.tplargs, self.out, os.environ
        ncu = b.props['multiprocessor_count']

        # The problem
        defs = {
            'LUINV_N': t['n'], 'LUINV_FTYPE': t['ftype'],
            'LUINV_STYPE': CTYPES[t['stype']],
            'LUINV_OTYPE': CTYPES[t['otype']],
            'LUINV_SCALE_T': npdtype_to_ctype(b.fpdtype),
            'LUINV_IXTYPE': {4: 'int', 8: 'int64_t'}[
                np.dtype(b.ixdtype).itemsize
            ],
            'LUINV_TR': out.trows, 'LUINV_TC': out.tcols,
            'LUINV_PADC': out.padc, 'LUINV_NDIES': b.ndies,
            'LUINV_FEW': int(t['nwork'] <= 3*ncu)
        }

        # Overrides of the knobs, and the instrumentation
        defs |= {m: int(env[k]) for k, m in KNOBS.items() if k in env}
        if env.get('LUINV_PHASES'):
            defs['LUINV_PHASES'] = 1

        # Extra macros for experiments, as LUINV_DEFS=A=1,B
        for d in filter(None, env.get('LUINV_DEFS', '').split(',')):
            k, _, v = d.partition('=')
            defs[k] = v or 1

        return RawHIPModule(b, HERE / 'lu_inv.hip', INCLUDE, defines=defs)

    @cached_property
    def waves(self):
        """Waves per block of the factorisation and the solves, from the
        launch bounds the kernels took."""
        return tuple(self.module.get_function(k, []).max_threads // 64
                     for k in (FACTOR, SOLVE))

    @cached_property
    def kerns(self):
        b, t, mod = self.backend, self.tplargs, self.module
        nb, (k1w, wpb) = t['npad'] // 16, self.waves
        sizes = {v.n for ps in self.parts.values() for _, v in ps}

        # The kernels for each number of items inverted at once
        kerns = {}
        for s in sizes:
            kerns[s] = [
                mod.kernel(BUILD, [b.ixdtype, b.fpdtype, PTR, b.ixdtype, PTR,
                                   PTR],
                           (-(-s // 64)*t['npad'], 1, 1), (256, 1, 1)),
                mod.kernel(FACTOR, [b.ixdtype, PTR, PTR, PTR, PTR],
                           (s, 1, 1), (64*k1w, 1, 1)),
                mod.kernel(SOLVE, [b.ixdtype] + [PTR]*5,
                           (s*-(-nb // wpb), 1, 1), (64*wpb, 1, 1))
            ]

        return kerns

    def invert(self, key, scale):
        for src, inv in self.parts[key]:
            ks = self.kerns[inv.n]
            ks[0].set_args(inv.n, scale, *src.kargs('ml'), self.wk, self.ws)
            ks[1].set_args(inv.n, self.wk, self.ws, self.perm, self.ticks)
            ks[2].set_args(inv.n, *inv.kargs('v'), self.perm, self.wk,
                           self.ws)

            self.backend.run_kernels(ks)
