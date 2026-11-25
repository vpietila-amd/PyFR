# lu_inv test and benchmark harness

Correctness tests, benchmarks and performance analysis for PyFR's batched
LU inverse (`LUInverse`, `lu_inv.mako`) on the HIP backend.
Candidates run through PyFR's real backend, exactly as the block-Jacobi
preconditioner calls them: staged Jacobians in `jstage`, `invert(colour,
scale)`, inverses out in the tiled matrix that `precondapply` reads.

Run everything from the repository root, or put it on `PYTHONPATH`.

## Variants

| Name | What it is |
|---|---|
| `baseline` | Frozen copy of `luinv.py` + `lu_inv.mako` (kernel `lu_inv_baseline`); the reference |
| `pyfr` | The live `pyfr.integrators.implicit.luinv.LUInverse` on this branch |
| `hip_ref` | The baseline kernel compiled as hand written HIP; bit-identical to `baseline` |
| `hip_ref_phases` | `hip_ref` with the time spent in each phase recorded |

Register new variants in `variants/__init__.py`. A variant is any class
with the interface of `LUInverse`: `__init__(backend, m, out, idxs,
extent)` and `invert(key, scale)`. Everything inside is free: threads,
blocks, grid, workspace, launches per call, number of kernels.

### Adding a hand written HIP candidate

```
variants/<name>/
  __init__.py                    # LUInverse subclass: kernels via RawHIPModule
  lu_inv.hip                     # entry: includes the headers, defines extern "C" kernels
  include/luinv/<name>/*.hpp     # device code; templates/constexpr, no Mako
```

Guard host-only includes with `#if !defined(__HIPCC_RTC__)` and avoid the
standard library in device code; HIPRTC supplies the HIP builtins itself.
`RawHIPModule(backend, entry, include, defines={...})` compiles the entry
with `-D` macros either through PyFR's HIPRTC compiler and cache (default;
as PyFR would) or offline with hipcc (`LUINV_COMPILER=hipcc`). Editing a
header rebuilds in both cases. Marshal arguments with `rawhip.kargs` or the
objects' own `kargs(form)`; see `variants/hip_ref` for a complete example.

## Correctness

```bash
cd bench/luinv
python -m pytest                            # quick suite, baseline, < 1 min
python -m pytest --variant hip_ref,pyfr     # other variants, or --variant all
python -m pytest -m full                    # whole matrix: n x precision x input kind
python -m pytest ../../pyfr/tests           # PyFR's own tests
```

Results are checked against `numpy.linalg.inv` with error bounds scaled to
the precision (see `check.py`), plus bug traps: colour isolation, coverage,
determinism and reuse with a new scale.

One case from the CLI:

```bash
python -m bench.luinv check --variant hip_ref --n 320 --precision single \
    --neles 1000 --ncolours 4 --kind pivot
```

## Benchmarks

```bash
python -m bench.luinv bench --n 320 --neles 4096 --reps 20     # one case
python -m bench.luinv sweep --variant baseline,hip_ref \
    --n 175,320,625 --precision double,single --neles 512,4096 --refs \
    --out bench/luinv/results/mysweep.jsonl
python -m bench.luinv compare bench/luinv/results/*.jsonl --ref baseline
python -m bench.luinv refs --n 320 --neles 4096                 # rocSOLVER, DGEMM, copy

cd bench/luinv
python -m pytest -m bench --variant hip_ref                     # benchmark suite
python -m pytest -m bench --bench-against results/old.jsonl --max-slowdown 0.03
```

Times are medians of HIP events around `invert()` on the backend's stream;
clocks are not locked. Each JSONL record carries the environment (git sha,
GPU, HIP version) and the resources of the kernels.

## Analysis

```bash
python -m bench.luinv resources --variant baseline --n 320   # hipcc remarks + ISA counts
python -m bench.luinv hipcc --variant hip_ref --n 320        # offline build, compile_commands.json
python -m bench.luinv counters --n 320 --neles 512           # rocprofv3 counters, derived metrics
python -m bench.luinv roofline --n 320 --neles 512 --kernel lu_inv_baseline
python -m bench.luinv phases --n 320 --neles 512             # time per phase of the kernel
```

- `counters` runs one rocprofv3 pass per counter group around the `profile`
  command and reports measured FLOPs, L2-to-fabric traffic, arithmetic
  intensity, L2 hit rate, stall and LDS figures.
- `roofline` runs `rocprof-compute profile` and `analyze`; the analysis
  needs its own venv for its pinned numpy/pandas
  (`LUINV_ROCPROF_PYTHON`, default `~/.venvs/rocprof-compute/bin/python`;
  create it with `pip install -r
  /opt/rocm/libexec/rocprofiler-compute/requirements.txt`).
- `phases` times the phases of the generated kernel through `hip_ref_phases`
  and gives their rates from the work model in `analysis/model.py`.

Build products go to `bench/luinv/build/` and results to
`bench/luinv/results/`; both are ignored by git.

## Notes

- `HIPEvent.elapsed_time` returns seconds.
- Cases above 8 GB of staged Jacobians need `--memory-model large`, as
  PyFR itself would; the kernel then uses 64-bit indices.
- The OpenMP backend needs libxsmm (`PYFR_XSMM_LIBRARY_PATH`); without it
  the OpenMP tests skip.
