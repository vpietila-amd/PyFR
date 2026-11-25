"""The LU inverse in separate kernels for the factorisation and the solves.

Where the factorisation is in fp64 or fp32 the kernels are hand written
with MFMA (see include/luinv/hip_v2): lu_build forms I - scale*src in a
strip-major workspace, lu_factor_mfma factorises it left-looking with a
block of four or eight waves per matrix, leaving the factor packed for the
MFMA, and lu_solve_mfma solves for 16 columns of the inverse per wave.  The
matrices are padded to a multiple of 16 for them.

Otherwise both kernels come from the rendering of the frozen baseline
template with the quick wins of hip_opt applied for the size, split at the
start of the solves: the factorisation takes one matrix per block and the
solves one column chunk of one matrix per block, each chunk with its own
right-hand side area in the workspace.

Options (environment variables, overriding the size policy):
  LUINV_PATH=<path>   mfma (the default) or gen
  LUINV_WPB=<n>       waves per block of the mfma solves, which share
                      their loads of the factor
  LUINV_K1_MINBLK=<n> launch bounds minimum blocks per CU of the mfma
                      factorisation
  LUINV_K1_WAVES=<n>  waves per block of the mfma factorisation
  LUINV_K2_MINBLK=<n> launch bounds minimum blocks per CU of the solves
  LUINV_NWORK=<n>     most matrices factorised at once on the mfma path
  LUINV_STRIPS=<n>    strips of 16 columns per pass of the factorisation
  LUINV_MINBLK=<n>    the same as LUINV_K2_MINBLK for the generated
                      factorisation
  LUINV_PIVOT=<mode>  generated pivot arg-max: shfl (default) or lds
  LUINV_IX32=0|1      generated: hoist the int64 workspace base out of the
                      indices
  LUINV_DEFS=A=1,B    extra macros for the mfma kernels, for experiments
"""
from functools import cached_property
from pathlib import Path
import os
import re

import numpy as np

from pyfr.nputil import npdtype_to_ctype

from bench.luinv.rawhip import PTR, RawHIPModule, hipcc_compatible
from bench.luinv.variants.baseline import LUInverse as BaselineLUInverse
from bench.luinv.variants.hip_opt import _extract_block, apply_opts


HERE = Path(__file__).parent
INCLUDE = HERE / 'include'
GENERATED = INCLUDE / 'luinv' / 'hip_v2' / 'generated'

TEMPLATE = 'bench.luinv.variants.baseline.lu_inv_baseline'

FACTOR, SOLVE = 'lu_factor_hip_v2', 'lu_solve_hip_v2'
BUILD, FACTOR_MFMA = 'lu_build_hip_v2', 'lu_factor_mfma_hip_v2'
SOLVE_MFMA = 'lu_solve_mfma_hip_v2'


def nchunk(tplargs):
    return -(-tplargs['npad'] // tplargs['sc'])


# C types of the hand written kernels for the storage types
CTYPES = {'double': 'double', 'float': 'float', 'half': '_Float16',
          'bf16': 'luinv::bf16'}


def mfma_path(t):
    return t['ftype'] in ('double', 'float')


def options(tplargs):
    """The options for the size, overridden by the environment."""
    env, t = os.environ, tplargs

    path = env.get('LUINV_PATH', 'mfma' if mfma_path(t) else 'gen')
    if path == 'mfma':
        nb = t['npad'] // 16

        # Blocks per CU as the registers allow, fp32 taking half as many,
        # and for big matrices two strips per pass of the factorisation and
        # eight waves to a block, cutting the latency of each matrix
        small, f32 = nb <= 20, t['ftype'] == 'float'
        k1minblk = (3 if f32 else 2) if small else 1
        return {
            'path': path,
            'wpb': int(env.get('LUINV_WPB', 4)),
            'k1minblk': int(env.get('LUINV_K1_MINBLK', k1minblk)),
            'k1w': int(env.get('LUINV_K1_WAVES', 4 if small else 8)),
            'strips': int(env.get('LUINV_STRIPS', 1 if small else 2)),
            'k2minblk': int(env.get('LUINV_K2_MINBLK', 2 if small else 1))
        }

    # Big matrices spill heavily at 2 blocks per CU
    big = t['npad'] > 512
    opts = {'path': 'gen', 'pivot': 'shfl', 'minblk': 1 if big else 2,
            'ix32': True}

    if 'LUINV_MINBLK' in env:
        opts['minblk'] = int(env['LUINV_MINBLK'])
    if 'LUINV_PIVOT' in env:
        opts['pivot'] = env['LUINV_PIVOT']
    if 'LUINV_IX32' in env:
        opts['ix32'] = env['LUINV_IX32'] not in ('', '0')

    # The solves spill less, so keep two blocks per CU for them
    opts['k2minblk'] = int(env.get('LUINV_K2_MINBLK', 2))

    return opts


def opt_tag(opts):
    return '_'.join(f'{k}{int(v) if isinstance(v, bool) else v}'
                    for k, v in sorted(opts.items()))


def _hoist(src, base):
    # ws_v[base + X] -> (ws_v + base)[X], keeping X a 32-bit offset
    return re.sub(rf'ws_v\[{base} \+ ([^\]]+)\]', rf'(ws_v + {base})[\1]',
                  src)


def split(src, tplargs, opts):
    """Split the rendered kernel into the factorisation and the solves."""
    NP, sc = tplargs['npad'], tplargs['sc']

    # The definitions ahead of the kernel, its signature and its storage
    i_sig = src.index('__global__')
    i_open = src.index('{', i_sig)
    xdecl = 'ixdtype_t _x = ixdtype_t(PYFR_BLOCK_ID);'
    i_x = src.index(xdecl)
    defs, sig, shared = src[:i_sig], src[i_sig:i_open], src[i_open + 1:i_x]

    # The factorisation runs from the workspace offsets up to the solves
    i_wo = src.index('// Offsets of the workspace')
    i_solve = src.index('// Solve LU X = I one column chunk at a time')
    factor = src[i_wo:i_solve]

    # The body of the loop over the column chunks
    s, e = _extract_block(src, '// Start from this chunk of the columns')
    chunk = src[s + 1:e - 1]

    # Give each chunk its own right-hand side area
    if f'wo + {NP*NP}' not in chunk:
        raise RuntimeError('right-hand side offset not found')
    chunk = chunk.replace(f'wo + {NP*NP}', 'ro')

    k2pre = f'''
    ixdtype_t _b = ixdtype_t(PYFR_BLOCK_ID);
    ixdtype_t _x = _b / {nchunk(tplargs)};
    int cc = int(_b % {nchunk(tplargs)})*{sc};
    ixdtype_t wo = wk_v[_x]*{tplargs['wsz']}, po = wk_v[_x]*{NP};
    ixdtype_t ro = wo + {NP*NP} + ixdtype_t(cc)*{NP};
'''

    lb = r'__launch_bounds__\(\s*(\d+)\s*,\s*\d+\s*\)'
    k1sig = re.sub(lb, rf'__launch_bounds__(\1, {opts["minblk"]})', sig)
    k2sig = re.sub(lb, rf'__launch_bounds__(\1, {opts["k2minblk"]})', sig)
    k1sig = k1sig.replace('void lu_inv_baseline(', f'void {FACTOR}(')
    k2sig = k2sig.replace('void lu_inv_baseline(', f'void {SOLVE}(')

    k1 = f'{k1sig}{{{shared}{xdecl}\n{{\n{factor}}}\n}}\n'
    k2 = f'{k2sig}{{{shared}{k2pre}{{\n{chunk}}}\n}}\n'

    if opts['ix32']:
        k1, k2 = _hoist(k1, 'wo'), _hoist(_hoist(k2, 'wo'), 'ro')

    return f'{defs}\n{k1}\n{k2}'


def header_name(tplargs, opts):
    return 'lu_inv_n{n}_{stype}_{ftype}_{otype}'.format(**tplargs) + \
        f'_{opt_tag(opts)}.hpp'


def generate(backend, tplargs, opts):
    src, *_ = backend.pointwise._render_kernel(
        'lu_inv_baseline', TEMPLATE, {}, tplargs, {}
    )
    src = hipcc_compatible(src)

    # The pivot search of hip_opt, applied before the split
    if opts['pivot'] != 'lds':
        src = apply_opts(src, {'pivot': opts['pivot']}, tplargs)

    hdr = ('// Generated by bench/luinv/variants/hip_v2 from the frozen '
           f'baseline\n// template with opts {opt_tag(opts)}; do not edit\n'
           '#pragma once\n' + split(src, tplargs, opts))

    path = GENERATED / header_name(tplargs, opts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != hdr:
        path.write_text(hdr)

    return path


class LUInverse(BaselineLUInverse):
    def _tplargs(self, m, out):
        t = super()._tplargs(m, out)

        if os.environ.get('LUINV_PATH', 'mfma') == 'mfma' and mfma_path(t):
            # Pad to a multiple of 16 for the hand written kernels, whose
            # workspace holds the packed factor and the inverses of the
            # diagonal blocks of U
            t['npad'] = npad = -(-t['n'] // 16)*16
            t['wsz'] = npad*(npad + 16)

            # Factorise more matrices at once, within a 2 GiB workspace
            nwork = int(os.environ.get('LUINV_NWORK', 0))
            nwork = nwork or max(512, min(4096, 2**28 // t["wsz"]))

            # Starting each part at a whole AoSoA group of the matrices
            t['nwork'] = min(nwork // 64 * 64, m.ioshape[2])
        else:
            # The factor, then a right-hand side area for each column chunk
            t['wsz'] = t['npad']*(t['npad'] + nchunk(t)*t['sc'])

        return t

    @cached_property
    def opts(self):
        return options(self.tplargs)

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
        b, t, opts, out = self.backend, self.tplargs, self.opts, self.out

        if opts['path'] == 'mfma':
            defs = {
                'LUINV_MFMA': 1, 'LUINV_N': t['n'], 'LUINV_NP': t['npad'],
                'LUINV_WSZ': f"{t['wsz']}LL", 'LUINV_WPB': opts['wpb'],
                'LUINV_K1MB': opts['k1minblk'], 'LUINV_K1W': opts['k1w'],
                'LUINV_S': opts['strips'],
                'LUINV_K2MB': opts['k2minblk'], 'LUINV_TR': out.trows,
                'LUINV_TC': out.tcols, 'LUINV_PADC': out.padc,
                'LUINV_NDIES': b.ndies, 'LUINV_FTYPE': t['ftype'],
                'LUINV_STYPE': CTYPES[t['stype']],
                'LUINV_OTYPE': CTYPES[t['otype']],
                'LUINV_SCALE_T': npdtype_to_ctype(b.fpdtype),
                'LUINV_IXTYPE': {4: 'int', 8: 'int64_t'}[
                    np.dtype(b.ixdtype).itemsize
                ]
            }
            if os.environ.get('LUINV_PHASES'):
                defs['LUINV_PHASES'] = 1

            # Extra macros for experiments, as LUINV_DEFS=A=1,B
            for d in filter(None, os.environ.get('LUINV_DEFS', '').split(',')):
                k, _, v = d.partition('=')
                defs[k] = v or 1
        else:
            hdr = generate(b, t, opts)
            defs = {'LUINV_HEADER': hdr.name}

        return RawHIPModule(b, HERE / 'lu_inv.hip', INCLUDE, defines=defs)

    @cached_property
    def kerns(self):
        b, t, mod = self.backend, self.tplargs, self.module
        sizes = {v.n for ps in self.parts.values() for _, v in ps}

        # The kernels for each number of items inverted at once
        kerns = {}
        for s in sizes:
            if self.opts['path'] == 'mfma':
                nb, wpb = t['npad'] // 16, self.opts['wpb']
                kerns[s] = [
                    mod.kernel(BUILD, [b.ixdtype, b.fpdtype, PTR, b.ixdtype,
                                       PTR, PTR],
                               (-(-s // 64)*t['npad'], 1, 1), (256, 1, 1)),
                    mod.kernel(FACTOR_MFMA, [b.ixdtype, PTR, PTR, PTR, PTR],
                               (s, 1, 1), (64*self.opts['k1w'], 1, 1)),
                    mod.kernel(SOLVE_MFMA, [b.ixdtype] + [PTR]*5,
                               (s*-(-nb // wpb), 1, 1), (64*wpb, 1, 1))
                ]
            else:
                argt = [b.ixdtype, b.fpdtype, PTR, PTR, PTR, PTR, b.ixdtype,
                        PTR, PTR]
                block = (t['nthreads'], 1, 1)
                kerns[s] = [
                    mod.kernel(FACTOR, argt, (s, 1, 1), block),
                    mod.kernel(SOLVE, argt, (s*nchunk(t), 1, 1), block)
                ]

        return kerns

    def invert(self, key, scale):
        for src, inv in self.parts[key]:
            ks = self.kerns[inv.n]

            if self.opts['path'] == 'mfma':
                ks[0].set_args(inv.n, scale, *src.kargs('ml'), self.wk,
                               self.ws)
                ks[1].set_args(inv.n, self.wk, self.ws, self.perm,
                               self.ticks)
                ks[2].set_args(inv.n, *inv.kargs('v'), self.perm, self.wk,
                               self.ws)
            else:
                args = (inv.n, scale, *inv.kargs('v'), self.perm,
                        *src.kargs('ml'), self.wk, self.ws)
                for k in ks:
                    k.set_args(*args)

            self.backend.run_kernels(ks)
