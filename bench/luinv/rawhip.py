import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from pyfr.backends.hip.compiler import HIPCompiler
from pyfr.backends.hip.provider import HIPKernel
from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import digest


# Offline hipcc builds with their ISA and resource usage remarks
BUILD_DIR = Path(__file__).parent / 'build' / 'hipcc'


def kargs(*objs):
    """Expand backend objects into raw kernel arguments.

    Each item is either a scalar, a backend object, or an (object, form)
    pair using the forms of PyFR's kernel argument specs; for example a
    matrix slice with 'ml' gives its pointer and leading dimension and a
    view gives its base pointer and mapping.
    """
    out = []
    for o in objs:
        if isinstance(o, tuple):
            out.extend(o[0].kargs(o[1]))
        else:
            out.append(o)

    return out


def hipcc_compatible(src):
    """Make the source of a generated PyFR kernel compile with hipcc too."""

    # hipcc gets the fixed width integer types from its own headers
    return re.sub(r'^(typedef [\w ]+ u?int\d+_t;)$',
                  r'#if defined(__HIPCC_RTC__)\n\1\n#endif', src,
                  flags=re.M)


class RawHIPKernel(HIPKernel):
    def __init__(self, fun, grid, block, sharedb=0):
        super().__init__()

        self.fun = fun
        self.params = fun.make_params(grid, block, sharedb)

    def set_args(self, *args):
        self.params.set_args(*args)

    def add_to_graph(self, graph, deps):
        return graph.graph.add_kernel(self.params, deps)

    def run(self, stream):
        self.fun.exec_async(stream, self.params)


class RawHIPModule:
    """Hand written HIP compiled for the backend.

    The source is a real .hip entry file which includes headers from its
    include directory; parameters arrive as -D macros.  It is compiled
    either by HIPRTC through the backend's compiler and cache, as PyFR does,
    or offline by hipcc with identical macros and include paths; the
    LUINV_COMPILER environment variable picks the default.  Editing a
    header rebuilds the module in both cases.
    """

    def __init__(self, backend, entry, include=None, defines={}, flags=[],
                 compiler=None):
        if backend.name != 'hip':
            raise RuntimeError('Raw HIP kernels need the HIP backend')

        self.backend = backend
        self.entry = Path(entry).resolve()
        self.include = Path(include or self.entry.parent).resolve()
        self.compiler = compiler or os.environ.get('LUINV_COMPILER',
                                                   'hiprtc')

        self.arch = backend.props['gcn_arch_name']
        self.flags = ['-munsafe-fp-atomics', f'-I{self.include}', *flags,
                      *(f'-D{k}={v}' for k, v in defines.items())]

        with backend.region('compile'):
            if self.compiler == 'hiprtc':
                code = backend.compiler.build(
                    self.entry.stem, self.entry.read_text(),
                    [f'--gpu-architecture={self.arch}', *self.flags]
                )
                self.artifacts = None
            elif self.compiler == 'hipcc':
                code = self.hipcc()
            else:
                raise ValueError(f'Unknown compiler {self.compiler!r}')

            self.mod = backend.hip.load_module(code)

    def hipcc_command(self, out):
        return ['hipcc', '--genco', f'--offload-arch={self.arch}', '-O3',
                *self.flags, '-save-temps=obj',
                '-Rpass-analysis=kernel-resource-usage', str(self.entry),
                '-o', str(out)]

    def hipcc(self):
        # Key the build on its command, source and every included header
        incs = HIPCompiler._include_files([f'-I{self.include}'])
        key = digest(self.hipcc_command('kernel.hsaco'),
                     self.entry.read_bytes(), incs)

        self.artifacts = adir = BUILD_DIR / f'{self.entry.stem}-{key[:16]}'
        out = adir / 'kernel.hsaco'

        if not out.exists():
            adir.mkdir(parents=True, exist_ok=True)
            cmd = self.hipcc_command(out)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode:
                raise RuntimeError(f'hipcc failed:\n{res.stderr}')

            (adir / 'remarks.txt').write_text(res.stderr)
            (adir / 'command.json').write_text(json.dumps(cmd))

        return out.read_bytes()

    def remarks(self):
        """Resource usage reported by hipcc for each kernel."""
        if not self.artifacts:
            return {}

        kerns, name = {}, None
        for l in (self.artifacts / 'remarks.txt').read_text().splitlines():
            if m := re.search(r'Function Name: (\w+)', l):
                name = m[1]
                kerns[name] = {}
            elif name and (m := re.search(r'remark: .*?:\d+:\d+:\s+'
                                          r'([^:]+): (\S+) \[', l)):
                kerns[name][m[1]] = m[2]

        return kerns

    def get_function(self, name, argtypes):
        argtypes = [npdtype_to_ctypestype(t) for t in argtypes]
        return self.mod.get_function(name, argtypes)

    def kernel(self, name, argtypes, grid, block, sharedb=0):
        fun = self.get_function(name, argtypes)
        return RawHIPKernel(fun, grid, block, sharedb)


# Argument type of pointers, as in PyFR's kernel generators
PTR = np.uintp
