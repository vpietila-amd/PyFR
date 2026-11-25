"""Optimisation playground variant for the staged hand-written experiments.

It renders the frozen baseline template, then applies source transforms for
the experiment knobs selected by the environment.  With no knobs set it
reproduces the baseline (and hip_ref) bit-for-bit, which is the Exp 0 parity
check.  Each knob is encoded in the generated header name and the compiled
kernel, so different settings do not collide in the build cache.

Knobs (environment variables):
  LUINV_MINBLK=<n>   Exp 1: rewrite the __launch_bounds__ minimum blocks per
                     CU (baseline is 2); 1 gives 512 VGPR at 1 block/CU.
  LUINV_PIVOT=shfl   Exp 2: replace the all-lane LDS butterfly of the pivot
                     arg-max with an intra-wave __shfl_xor reduction plus one
                     cross-wave LDS step, and drop the LDS-only threadfence.
  LUINV_IX32=1       Exp 4: hoist the int64 workspace base out of the global
                     indices (ws_v[wo + X] -> (ws_v + wo)[X]) so the per-matrix
                     offsets stay 32-bit; only bites in the large memory model
                     (n=625 x 4096, where wk offsets exceed int32).
  LUINV_FENCE=lds    Exp 5: redefine the cooperative barrier to drop the
                     __threadfence_block, i.e. plain __syncthreads everywhere.
                     Upper-bound probe for removing the global fence from the
                     LDS-only staging barriers (not a safe keeper as-is: the
                     ws-cooperative barriers still need the fence).
"""
from functools import cached_property
from pathlib import Path
import os
import re

from bench.luinv.rawhip import PTR, RawHIPModule, hipcc_compatible
from bench.luinv.variants.baseline import LUInverse as BaselineLUInverse


HERE = Path(__file__).parent
INCLUDE = HERE / 'include'
GENERATED = INCLUDE / 'luinv' / 'hip_opt' / 'generated'

# The frozen baseline template, shared with hip_ref
TEMPLATE = 'bench.luinv.variants.baseline.lu_inv_baseline'


def active_opts():
    """Collect the experiment knobs set in the environment."""
    opts = {}

    minblk = os.environ.get('LUINV_MINBLK')
    if minblk:
        opts['minblk'] = int(minblk)

    pivot = os.environ.get('LUINV_PIVOT')
    if pivot:
        opts['pivot'] = pivot

    if os.environ.get('LUINV_IX32'):
        opts['ix32'] = 'on'

    fence = os.environ.get('LUINV_FENCE')
    if fence:
        opts['fence'] = fence

    return opts


def _extract_block(src, anchor):
    """Return (start, end) spanning the brace block wrapping anchor.

    `anchor` is a substring inside the block; the block is the nearest `{`
    before it through its brace-balanced close.
    """
    i = src.find(anchor)
    if i == -1 or src.find(anchor, i + len(anchor)) != -1:
        raise RuntimeError(f'expected exactly one {anchor!r}')

    start = src.rfind('{', 0, i)
    depth = 0
    for j in range(start, len(src)):
        if src[j] == '{':
            depth += 1
        elif src[j] == '}':
            depth -= 1
            if depth == 0:
                return start, j + 1

    raise RuntimeError('unbalanced braces')


def _pivot_shfl(ftype, nthreads):
    """A hierarchical arg-max replacing the all-lane LDS butterfly."""
    if nthreads % 64:
        raise RuntimeError('pivot=shfl needs a multiple of 64 lanes')
    nwave = nthreads // 64
    cmp = '(mx_o > mx || (mx_o == mx && mi_o < mi))'

    return f'''{{
            // Exp 2: intra-wave arg-max by shuffles, then one cross-wave step
            for (int _m = 32; _m >= 1; _m >>= 1)
            {{
                {ftype} mx_o = __shfl_xor(mx, _m);
                int mi_o = __shfl_xor(mi, _m);
                if {cmp} {{ mx = mx_o; mi = mi_o; }}
            }}
            if (((int)threadIdx.x & 63) == 0)
            {{
                _rs_{ftype}_0[(int)threadIdx.x >> 6] = mx;
                _rs_int_0[(int)threadIdx.x >> 6] = mi;
            }}
            __syncthreads();
            #pragma unroll
            for (int _w = 0; _w < {nwave}; _w++)
            {{
                {ftype} mx_o = _rs_{ftype}_0[_w];
                int mi_o = _rs_int_0[_w];
                if {cmp} {{ mx = mx_o; mi = mi_o; }}
            }}
            __syncthreads();
        }}'''


def apply_opts(src, opts, tplargs):
    """Apply the selected experiment transforms to the rendered source."""
    # Exp 1: occupancy / spill -- set the launch_bounds minimum blocks per CU
    if 'minblk' in opts:
        src, n = re.subn(r'(__launch_bounds__\(\s*\d+\s*,\s*)\d+(\s*\))',
                         rf'\g<1>{opts["minblk"]}\g<2>', src, count=1)
        if n != 1:
            raise RuntimeError('could not rewrite __launch_bounds__')

    # Exp 2: replace the pivot arg-max butterfly with a shuffle reduction
    if opts.get('pivot') == 'shfl':
        start, end = _extract_block(src, 'for (int _m =')
        block = _pivot_shfl(tplargs['ftype'], tplargs['nthreads'])
        src = src[:start] + block + src[end:]
    elif 'pivot' in opts:
        raise RuntimeError(f'unknown pivot mode {opts["pivot"]!r}')

    # Exp 4: hoist the int64 base pointer so the per-matrix offsets stay int32
    if 'ix32' in opts:
        src, nw = re.subn(r'ws_v\[wo \+ ([^\]]+)\]', r'(ws_v + wo)[\1]', src)
        src, np_ = re.subn(r'perm_v\[po \+ ([^\]]+)\]',
                           r'(perm_v + po)[\1]', src)
        if nw == 0 or np_ == 0:
            raise RuntimeError('ix32: no ws_v[wo+..]/perm_v[po+..] sites')

    # Exp 5: drop the __threadfence_block from the cooperative barrier macro
    if opts.get('fence') == 'lds':
        old = ('#define PYFR_SYNC_GMEM_THREADS() do { __threadfence_block(); '
               '__syncthreads(); } while (0)')
        new = '#define PYFR_SYNC_GMEM_THREADS() do { __syncthreads(); } while (0)'
        src, nf = src.replace(old, new), src.count(old)
        if nf != 1:
            raise RuntimeError('fence: expected one SYNC_GMEM macro definition')
    elif 'fence' in opts:
        raise RuntimeError(f'unknown fence mode {opts["fence"]!r}')

    return src


def opt_tag(opts):
    """A stable suffix naming the active options, or 'base' when none."""
    return '_'.join(f'{k}{v}' for k, v in sorted(opts.items())) or 'base'


def header_name(tplargs, opts):
    return 'lu_inv_n{n}_{stype}_{ftype}_{otype}'.format(**tplargs) + \
        f'_{opt_tag(opts)}.hpp'


def generate(backend, tplargs, opts):
    """Render the baseline kernel for the HIP route with the opts applied."""
    src, *_ = backend.pointwise._render_kernel(
        'lu_inv_baseline', TEMPLATE, {}, tplargs, {}
    )

    src = hipcc_compatible(src)
    src = src.replace('void lu_inv_baseline(', 'void lu_inv_hip_opt(')
    src = apply_opts(src, opts, tplargs)

    hdr = ('// Generated by bench/luinv/variants/hip_opt from the frozen '
           f'baseline\n// template with opts {opt_tag(opts)}; do not edit\n'
           '#pragma once\n' + src)

    path = GENERATED / header_name(tplargs, opts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != hdr:
        path.write_text(hdr)

    return path


class LUInverse(BaselineLUInverse):
    @cached_property
    def opts(self):
        return active_opts()

    @cached_property
    def module(self):
        hdr = generate(self.backend, self.tplargs, self.opts)
        return RawHIPModule(self.backend, HERE / 'lu_inv.hip', INCLUDE,
                            defines={'LUINV_HEADER': hdr.name})

    @cached_property
    def kerns(self):
        b = self.backend
        argt = [b.ixdtype, b.fpdtype, PTR, PTR, PTR, PTR, b.ixdtype, PTR,
                PTR]
        block = (self.tplargs['nthreads'], 1, 1)

        # One kernel for each number of items inverted at once
        sizes = {v.n for ps in self.parts.values() for _, v in ps}
        return {s: self.module.kernel('lu_inv_hip_opt', argt, (s, 1, 1),
                                      block)
                for s in sizes}

    def invert(self, key, scale):
        for src, inv in self.parts[key]:
            kern = self.kerns[inv.n]
            kern.set_args(inv.n, scale, *inv.kargs('v'), self.perm,
                          *src.kargs('ml'), self.wk, self.ws)
            self.backend.run_kernels([kern])


def variant_with_opts(opts):
    """A LUInverse subclass whose opts are fixed, not read from the
    environment.  Lets one process compile and compare several knob
    configurations, as the tuning search does; the environment path on the
    plain ``LUInverse`` is unchanged.
    """
    fixed = dict(opts)

    class _LUInverse(LUInverse):
        @cached_property
        def opts(self):
            return dict(fixed)

    return _LUInverse
