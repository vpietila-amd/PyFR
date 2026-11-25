import datetime
import json
from pathlib import Path
import platform
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git():
    status = _run('git', 'status', '--porcelain', '--untracked-files=no')
    return {'sha': _run('git', 'rev-parse', 'HEAD'),
            'branch': _run('git', 'rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty': bool(status) if status is not None else None}


def _gpu(backend):
    if backend.name != 'hip':
        return {}

    props = backend.props
    info = {'gpu': props['name'], 'arch': props['gcn_arch_name'],
            'ncu': props['multiprocessor_count'],
            'hip_runtime': backend.compiler.version}

    # Clock and power limits; the clocks themselves are not locked
    smi = _run('amd-smi', 'metric', '-g', '0', '-c', '-p', '--json')
    try:
        dev = json.loads(smi)['gpu_data'][0]
        clk = dev['clock']['gfx_0']
        info['gfx_max_clk_mhz'] = clk['max_clk']['value']
        info['clk_locked'] = clk['clk_locked']
        info['power_management'] = dev['power']['power_management']
    except (TypeError, ValueError, KeyError, IndexError):
        pass

    return info


def environment(backend):
    return {
        'timestamp': datetime.datetime.now().astimezone().isoformat(),
        'host': platform.node(),
        'numpy': np.__version__,
        'backend': backend.name,
        'precision': np.dtype(backend.fpdtype).name,
        'git': _git(),
        **_gpu(backend),
    }
