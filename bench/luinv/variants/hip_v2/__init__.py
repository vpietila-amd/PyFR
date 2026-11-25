"""The LU inverse split into a factorisation and a solve kernel.

The factorisation kernel takes one matrix per block, as the baseline does,
and leaves the factor and row order in the workspace.  The solve kernel
then takes one column chunk of one matrix per block, so the solves, the
larger part of the work, run with as many blocks as there are chunks.  Each
chunk has its own right-hand side area in the workspace.

Both kernels come from the rendering of the frozen baseline template with
the quick wins of hip_opt applied for the size, split at the start of the
solves.  Options (environment variables, overriding the size policy):
  LUINV_MINBLK=<n>    launch bounds minimum blocks per CU of both kernels
  LUINV_K2_MINBLK=<n> the same for the solve kernel only
  LUINV_PIVOT=<mode>  pivot arg-max: shfl (default) or lds (the baseline)
  LUINV_IX32=0|1      hoist the int64 workspace base out of the indices
  LUINV_K2=<kind>     solves: mfma (hand written, the default where it
                      applies: fp64 with npad a multiple of 16) or gen
  LUINV_WPB=<n>       waves per block of the mfma solves
"""
from functools import cached_property
from pathlib import Path
import os
import re

from bench.luinv.rawhip import PTR, RawHIPModule, hipcc_compatible
from bench.luinv.variants.baseline import LUInverse as BaselineLUInverse
from bench.luinv.variants.hip_opt import _extract_block, apply_opts


HERE = Path(__file__).parent
INCLUDE = HERE / 'include'
GENERATED = INCLUDE / 'luinv' / 'hip_v2' / 'generated'

TEMPLATE = 'bench.luinv.variants.baseline.lu_inv_baseline'

FACTOR, SOLVE = 'lu_factor_hip_v2', 'lu_solve_hip_v2'
PACK, SOLVE_MFMA = 'lu_pack_hip_v2', 'lu_solve_mfma_hip_v2'


def nchunk(tplargs):
    return -(-tplargs['npad'] // tplargs['sc'])


def options(tplargs):
    """The options for the size, overridden by the environment."""
    # Big matrices spill heavily at 2 blocks per CU
    big = tplargs['npad'] > 512
    opts = {'pivot': 'shfl', 'minblk': 1 if big else 2, 'ix32': True}

    env = os.environ
    if 'LUINV_MINBLK' in env:
        opts['minblk'] = int(env['LUINV_MINBLK'])

    # The solves spill less, so keep two blocks per CU for them
    opts['k2minblk'] = int(env.get('LUINV_K2_MINBLK', 2))
    if 'LUINV_PIVOT' in env:
        opts['pivot'] = env['LUINV_PIVOT']
    if 'LUINV_IX32' in env:
        opts['ix32'] = env['LUINV_IX32'] not in ('', '0')

    # The MFMA solves take fp64 in 16 x 16 tiles
    nb, mfma = tplargs['npad'] // 16, tplargs['npad'] % 16 == 0
    mfma &= tplargs['ftype'] == tplargs['otype'] == 'double'
    opts['k2'] = env.get('LUINV_K2', 'mfma' if mfma else 'gen')
    if opts['k2'] == 'mfma':
        wpb = next(w for w in (4, 2, 1) if nb % w == 0)
        opts['wpb'] = int(env.get('LUINV_WPB', wpb))

        # Two waves per SIMD while the tiles of Z fit in half the registers
        opts['k2minblk'] = int(env.get('LUINV_K2_MINBLK',
                                       2 if nb <= 20 else 1))

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

        # The factor, then a right-hand side area for each column chunk or
        # the packed factor and inverses of the diagonal blocks of U
        t['wsz'] = t['npad']*(t['npad'] + nchunk(t)*t['sc'] + 16)
        return t

    @cached_property
    def opts(self):
        return options(self.tplargs)

    @cached_property
    def module(self):
        t, opts, out = self.tplargs, self.opts, self.out
        hdr = generate(self.backend, t, opts)

        defs = {'LUINV_HEADER': hdr.name}
        if opts['k2'] == 'mfma':
            defs |= {
                'LUINV_MFMA': 1, 'LUINV_N': t['n'], 'LUINV_NP': t['npad'],
                'LUINV_WSZ': f"{t['wsz']}LL", 'LUINV_WPB': opts['wpb'],
                'LUINV_K2MB': opts['k2minblk'], 'LUINV_TR': out.trows,
                'LUINV_TC': out.tcols, 'LUINV_PADC': out.padc
            }

        return RawHIPModule(self.backend, HERE / 'lu_inv.hip', INCLUDE,
                            defines=defs)

    @cached_property
    def kerns(self):
        b = self.backend
        argt = [b.ixdtype, b.fpdtype, PTR, PTR, PTR, PTR, b.ixdtype, PTR,
                PTR]
        block = (self.tplargs['nthreads'], 1, 1)
        nc = nchunk(self.tplargs)

        mod, sizes = self.module, {v.n for ps in self.parts.values()
                                   for _, v in ps}

        # The kernels for each number of items inverted at once
        kerns = {}
        for s in sizes:
            ks = [mod.kernel(FACTOR, argt, (s, 1, 1), block)]
            if self.opts['k2'] == 'mfma':
                nb, wpb = self.tplargs['npad'] // 16, self.opts['wpb']
                ks.append(mod.kernel(PACK, [b.ixdtype, PTR, PTR],
                                     (s*nb, 1, 1), (256, 1, 1)))
                ks.append(mod.kernel(SOLVE_MFMA, [b.ixdtype] + [PTR]*5,
                                     (s*nb // wpb, 1, 1), (64*wpb, 1, 1)))
            else:
                ks.append(mod.kernel(SOLVE, argt, (s*nc, 1, 1), block))

            kerns[s] = ks

        return kerns

    def invert(self, key, scale):
        for src, inv in self.parts[key]:
            ks = self.kerns[inv.n]
            args = (inv.n, scale, *inv.kargs('v'), self.perm,
                    *src.kargs('ml'), self.wk, self.ws)
            ks[0].set_args(*args)

            if self.opts['k2'] == 'mfma':
                ks[1].set_args(inv.n, self.wk, self.ws)
                ks[2].set_args(inv.n, *inv.kargs('v'), self.perm, self.wk,
                               self.ws)
            else:
                ks[1].set_args(*args)

            self.backend.run_kernels(ks)
