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

PyFR itself is not baked in.

- **Your home directory is bind-mounted at the same absolute path.** Every
  host path is valid inside the container too: the checkout, scratch
  directories, other repos.
- The workspace folder is the checkout's own path, and PyFR is installed from
  it in editable mode.
- The container user's `$HOME` is still `/home/<user>` inside the image. This
  keeps the container's VS Code server and caches out of your real home,
  where they would clash with the host's `~/.vscode-server`.
- The checkout must be inside your home directory.

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
    -v $HOME:$HOME -w $PWD pyfr-hip-dev

# first time, inside the container (from the checkout)
pip install -e . --no-deps
```
