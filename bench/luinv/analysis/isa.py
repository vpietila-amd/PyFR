from pathlib import Path
import re


# Static instruction counts of interest; categories may overlap
CATEGORIES = {
    'barrier': r's_barrier',
    'waitcnt': r's_waitcnt',
    'mfma': r'v_mfma',
    'fma_f64': r'v_fmac?_f64',
    'valu_f64': r'v_\w+_f64',
    'fma_f32': r'v_fmac?_f32|v_pk_fma_f32',
    'valu': r'v_',
    'salu': r's_',
    'lds': r'ds_',
    'global_load': r'global_load',
    'global_store': r'global_store',
    'scratch_load': r'scratch_load|buffer_load',
    'scratch_store': r'scratch_store|buffer_store',
}


def kernel_isa(sfile, name):
    """The instructions of a kernel in a hipcc -save-temps assembly file."""
    body, lines = False, []
    for l in Path(sfile).read_text().splitlines():
        if l.startswith(f'{name}:'):
            body = True
        elif body and l.startswith('.Lfunc_end'):
            break
        elif body and re.match(r'\s+[a-z]', l):
            lines.append(l.strip())

    return lines


def isa_stats(sfile, name):
    insts = kernel_isa(sfile, name)
    mnems = [i.split()[0] for i in insts]

    stats = {'total': len(mnems)}
    for cat, pat in CATEGORIES.items():
        stats[cat] = sum(1 for m in mnems if re.match(pat, m))

    # Cross lane operations through DPP or LDS permutes
    stats['dpp'] = sum(1 for i in insts if re.search(r'row_|quad_perm', i))
    stats['permute'] = sum(1 for m in mnems if re.match(r'ds_b?permute|'
                                                     r'ds_swizzle', m))

    return stats


def assembly_file(artifacts):
    # The device assembly saved by -save-temps
    return next(Path(artifacts).glob('*amdgcn*.s'), None) or \
        next(Path(artifacts).glob('*gfx*.s'))
