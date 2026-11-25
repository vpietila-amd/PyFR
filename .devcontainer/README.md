# PyFR HIP development container

A ROCm 10.2 (`rocm/dev-ubuntu-24.04:10.2.0-full`) image for developing and
profiling PyFR's HIP backend kernels on AMD Instinct GPUs (gfx942 by default,
`--build-arg GPU_ARCH=gfx950` for MI350/MI355).

The image contains:

- **ROCm tools:** hipcc/amdclang++, HIPRTC, rocBLAS and roctx (the libraries
  PyFR's HIP backend loads), rocgdb, rocprofv3, rocprof-compute and
  rocprof-sys, plus llvm-objdump and llvm-mc for ISA inspection.
- **Build tools:** cmake, ninja, git, ccache, OpenMPI and HDF5.
- **Python:** a virtualenv in `/opt/venv` holding PyFR's runtime dependencies,
  pytest and ipython.

PyFR itself is not baked in. The checkout is bind-mounted at `/workspace/PyFR`
and installed in editable mode.

## VS Code

Run *Dev Containers: Reopen in Container*. If your host's `render` group is not
GID 109, export `RENDER_GID=$(getent group render | cut -d: -f3)` before
launching VS Code.

## Command line

```bash
docker build -f .devcontainer/Dockerfile \
    --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USERNAME=$(id -un) \
    --build-arg RENDER_GID=$(getent group render | cut -d: -f3) \
    -t pyfr-hip-dev .devcontainer

docker run --rm -it --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --ipc=host --init \
    -v $PWD:/workspace/PyFR pyfr-hip-dev

# first time, inside the container
pip install -e /workspace/PyFR --no-deps
```
