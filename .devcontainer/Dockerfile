# PyFR HIP-backend / lu_inv kernel development container (MI300X gfx942, or gfx950).
#
#   docker build -f .devcontainer/Dockerfile \
#       --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USERNAME=$(id -un) \
#       --build-arg RENDER_GID=$(getent group render | cut -d: -f3) \
#       -t pyfr-hip-dev .devcontainer
#
# The image carries dependencies only.  The host home directory is
# bind-mounted at the same absolute path (see devcontainer.json / README.md)
# and PyFR is installed editable from the checkout on first run
# (`pip install -e <checkout> --no-deps`).
#
# Base: rocm/dev-ubuntu-24.04:10.2.0-full (ROCm 10.2.0, released 2026-11-18, the
# newest tag on Docker Hub). Verified in the image: hipcc, amdclang++,
# llvm-objdump/llvm-mc/llvm-readelf (/opt/rocm/lib/llvm/bin), rocgdb,
# rocprofv3, rocprof-compute, rocprof-sys-*, libhiprtc, librocblas,
# librocprofiler-sdk-roctx (all that PyFR's HIP backend dlopens), Python 3.12.3.
# NOT in the base: cmake, ninja, git, MPI, roc-obj-* wrappers (use llvm-objdump).
ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:10.2.0-full
FROM ${BASE_IMAGE}

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
# GID of the host's `render` group (owner of /dev/kfd); see below.
ARG RENDER_GID=109
# gfx942 = MI300X, gfx950 = MI350/MI355. Space/semicolon lists are fine for
# hipcc (--offload-arch); PYFR itself targets the device it finds at runtime.
ARG GPU_ARCH=gfx942

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ccache pkg-config \
      gdb jq less vim-tiny tmux ca-certificates \
      python3-venv python3-dev \
      libopenmpi-dev openmpi-bin \
      libhdf5-dev libhdf5-openmpi-dev \
 && rm -rf /var/lib/apt/lists/*

# Non-root user matching the host UID/GID. Ubuntu 24.04 images ship an
# `ubuntu` user at uid/gid 1000 -- drop it so the host IDs can be reused.
# `useradd -l` avoids a huge sparse /var/log/lastlog with large (AD/LDAP) UIDs.
#
# /dev/kfd and /dev/dri/renderD* belong to the host's `render` group, so the
# container's `render` group must have the same GID for `--group-add=render`
# to grant GPU access (find yours with `getent group render`).
RUN if getent passwd ubuntu >/dev/null && [ "$(id -u ubuntu)" = "${UID}" ]; then userdel -r ubuntu; fi \
 && if getent group ${GID} >/dev/null; then groupdel $(getent group ${GID} | cut -d: -f1) 2>/dev/null || true; fi \
 && groupadd -g ${GID} ${USERNAME} \
 && useradd -l -u ${UID} -g ${GID} -m -s /bin/bash ${USERNAME} \
 && if getent group render >/dev/null; then groupmod -o -g ${RENDER_GID} render; \
    else groupadd -o -g ${RENDER_GID} render; fi \
 && usermod -aG video,render ${USERNAME} \
 && mkdir -p /workspace /home/${USERNAME}/.cache/pyfr /home/${USERNAME}/.vscode-server \
 && chown -R ${UID}:${GID} /workspace /home/${USERNAME}

# Python venv with PyFR's runtime deps (pyproject.toml [project].dependencies).
# h5py is built against system parallel-capable HDF5 only if you set
# HDF5_MPI=ON; the default wheel is fine for PyFR.
RUN python3 -m venv /opt/venv && chown -R ${UID}:${GID} /opt/venv
# rocprof-compute analyze pins numpy/pandas versions that clash with PyFR's,
# so it gets a venv of its own (used by bench/luinv/analysis/roofline.py).
RUN python3 -m venv /opt/rocprof-compute-venv \
 && chown -R ${UID}:${GID} /opt/rocprof-compute-venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/lib/llvm/bin:${PATH}
USER ${UID}:${GID}
RUN pip install \
      "numpy>=2.4.2" "mako>=1.0.0" "h5py>=2.10" "gimmik>=4.0" "boostree>=0.3.0" \
      "platformdirs>=2.2.0" "mpi4py>=4.0.0" \
      pytest pytest-xdist ipython \
      scipy pandas matplotlib
RUN /opt/rocprof-compute-venv/bin/pip install \
      -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
ENV LUINV_ROCPROF_PYTHON=/opt/rocprof-compute-venv/bin/python

# ROCm / PyFR environment. PyFR dlopens libamdhip64, libhiprtc, librocblas and
# librocprofiler-sdk-roctx by soname, so /opt/rocm/lib must be on the loader path.
ENV ROCM_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    HIP_PATH=/opt/rocm \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    GPU_ARCH=${GPU_ARCH} \
    HCC_AMDGPU_TARGET=${GPU_ARCH} \
    CCACHE_DIR=/home/${USERNAME}/.cache/ccache \
    XDG_CACHE_HOME=/home/${USERNAME}/.cache \
    OMPI_MCA_rmaps_base_oversubscribe=1

WORKDIR /workspace
CMD ["/bin/bash"]
