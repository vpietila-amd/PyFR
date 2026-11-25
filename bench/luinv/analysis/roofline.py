"""Full rocprof-compute profiles, with empirical rooflines, of a case."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from bench.luinv.env import ROOT


def _rocm_version():
    # The container lacks /opt/rocm/.info, which rocprof-compute reads
    out = subprocess.run(['amd-smi', 'version'], capture_output=True,
                         text=True).stdout
    m = re.search(r'ROCm version: ([\d.]+)', out)
    return m[1] if m else None


def _run(cmd, log):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    if not Path('/opt/rocm/.info/version').exists() and _rocm_version():
        env.setdefault('ROCM_VER', _rocm_version())

    with open(log, 'w') as f:
        res = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f,
                             stderr=subprocess.STDOUT, text=True)
    if res.returncode:
        raise RuntimeError(f'{cmd[0]} failed; see {log}')


# Analysis pins its own numpy and pandas, so it runs from a separate venv
ANALYZE_PYTHON = Path(os.environ.get(
    'LUINV_ROCPROF_PYTHON', Path.home() / '.venvs/rocprof-compute/bin/python'
))


def profile(case_argv, workdir, name, kernel=None, roof_only=False):
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    cmd = ['rocprof-compute', 'profile', '--name', name,
           '--output-directory', str(workdir), '--overwrite']
    if kernel:
        cmd += ['-k', kernel]
    if roof_only:
        cmd += ['--roof-only']

    # rocprof-compute scans the workload's arguments too, so avoid -m
    script = ROOT / 'bench' / 'luinv' / '__main__.py'
    cmd += ['--', sys.executable, str(script), 'profile', *case_argv]

    # Profiling with --overwrite empties the workload directory
    _run(cmd, workdir.with_suffix('.profile.log'))
    return workdir


def analyze(workdir, blocks=None):
    workdir = Path(workdir)
    cmd = ['rocprof-compute', 'analyze', '-p', str(workdir)]
    if ANALYZE_PYTHON.exists():
        cmd = [str(ANALYZE_PYTHON), shutil.which('rocprof-compute'), *cmd[1:]]
    if blocks:
        cmd += ['-b', *blocks]

    out = workdir / ('analyze.txt' if not blocks else
                     f"analyze-{'_'.join(blocks)}.txt")
    _run(cmd, out)
    return out
