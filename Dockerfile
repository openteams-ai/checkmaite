# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

ARG UBUNTU_IMAGE=docker.io/library/ubuntu:24.04@sha256:f610ab94648195aa356059f5b41d6085c9d4d903c072430cdd1af7bdb646106b
ARG CUDA_UBUNTU_IMAGE=docker.io/nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@sha256:4d242f206abc4b9588a6506cce2d88932cc879849395aae3785075179718cc49
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.24@sha256:816fdce3387ed2142e37d2e56e1b1b97ccc1ea87731ba199dc8a25c04e4997c5

# Base digests do not pin packages downloaded later from a moving archive.
# Every apt operation therefore resolves against this one Ubuntu archive
# snapshot (docker/snapshot_apt.sh), so a rebuild sees the same package versions
# until the timestamp is deliberately moved forward. Both products use Ubuntu
# 24.04: UBI 9 and 10 carry HIGH findings that Red Hat has not fixed. The
# refresh procedure is in docs/development/container_maintenance.md.
ARG UBUNTU_SNAPSHOT=20261002T000000Z

FROM ${UV_IMAGE} AS uv

# Build tooling shared by requirement export and the two project wheels. No
# build tooling is copied into either product container.
FROM ${UBUNTU_IMAGE} AS build-base
ARG UBUNTU_SNAPSHOT
ENV UV_PYTHON_DOWNLOADS=never
RUN --mount=type=bind,source=docker/snapshot_apt.sh,target=/usr/local/bin/snapshot-apt,ro \
    /usr/local/bin/snapshot-apt "${UBUNTU_SNAPSHOT}" python3.12 python3.12-venv
COPY --from=uv /uv /usr/local/bin/uv

# Export only from dependency metadata so application source changes do not
# invalidate either requirements file or the expensive dependency layers.
FROM build-base AS requirements
WORKDIR /build/docker/runtime
COPY pyproject.toml README.md LICENSE.md /build/
COPY docker/runtime/pyproject.toml docker/runtime/uv.lock ./
RUN UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
    uv export --quiet --locked --extra cpu --no-dev --no-emit-local \
        --output-file /requirements-cpu.txt \
    && uv export --quiet --locked --extra cuda --no-dev --no-emit-local \
        --output-file /requirements-cuda.txt

FROM build-base AS builder
ARG CHECKMAITE_VERSION=0.0.0
ARG TARGETARCH
WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE.md ./
COPY docker/runtime/pyproject.toml docker/runtime/uv.lock ./docker/runtime/
COPY docker/runtime/src ./docker/runtime/src
COPY src ./src
RUN --mount=type=cache,id=checkmaite-uv-build-${TARGETARCH},target=/root/.cache/uv,sharing=locked \
    UV_DYNAMIC_VERSIONING_BYPASS="${CHECKMAITE_VERSION}" \
        uv build --quiet --wheel --out-dir /wheels . \
    && uv build --quiet --wheel --out-dir /wheels docker/runtime

# Install dependencies separately from project wheels so source-only changes
# retain the expensive locked dependency layers.
FROM build-base AS install-base
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN uv venv --python /usr/bin/python3.12 /opt/venv

FROM install-base AS cpu-dependencies
ARG PYTORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
ARG TARGETARCH
COPY --from=requirements /requirements-cpu.txt /requirements.txt
RUN --mount=type=cache,id=checkmaite-uv-${TARGETARCH},target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=docker/install_locked_environment.sh,target=/usr/local/bin/install-locked-environment,ro \
    /usr/local/bin/install-locked-environment \
        /requirements.txt "${PYTORCH_CPU_INDEX}"

FROM cpu-dependencies AS cpu-environment
COPY --from=builder /wheels /wheels
RUN --mount=type=bind,source=docker/install_project_wheels.sh,target=/usr/local/bin/install-project-wheels,ro \
    /usr/local/bin/install-project-wheels /wheels

FROM install-base AS cuda-dependencies
ARG PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130
ARG TARGETARCH
COPY --from=requirements /requirements-cuda.txt /requirements.txt
RUN --mount=type=cache,id=checkmaite-uv-${TARGETARCH},target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=docker/install_locked_environment.sh,target=/usr/local/bin/install-locked-environment,ro \
    /usr/local/bin/install-locked-environment \
        /requirements.txt "${PYTORCH_CUDA_INDEX}"

FROM cuda-dependencies AS cuda-environment
COPY --from=builder /wheels /wheels
RUN --mount=type=bind,source=docker/install_project_wheels.sh,target=/usr/local/bin/install-project-wheels,ro \
    /usr/local/bin/install-project-wheels /wheels

# Runtime filesystems: the pinned Ubuntu (CPU) or NVIDIA Ubuntu (CUDA) image,
# brought to the snapshot, with Python added and package managers, unneeded
# tools, setuid bits, and ensurepip removed. dpkg metadata stays readable by
# scanners. The shared script keeps CPU and CUDA pruning identical.
FROM ${UBUNTU_IMAGE} AS cpu-rootfs
ARG UBUNTU_SNAPSHOT
RUN --mount=type=bind,source=docker/snapshot_apt.sh,target=/usr/local/bin/snapshot-apt,ro \
    --mount=type=bind,source=docker/prepare_runtime_rootfs.sh,target=/usr/local/bin/prepare-runtime-rootfs,ro \
    /usr/local/bin/snapshot-apt "${UBUNTU_SNAPSHOT}" python3.12 libstdc++6 ca-certificates \
    && /usr/local/bin/prepare-runtime-rootfs

# NVIDIA's CUDA, cuDNN, NCCL, driver constraints, and runtime ENV are kept as
# published; only Ubuntu packages move to the snapshot.
FROM ${CUDA_UBUNTU_IMAGE} AS cuda-rootfs
ARG UBUNTU_SNAPSHOT
RUN --mount=type=bind,source=docker/snapshot_apt.sh,target=/usr/local/bin/snapshot-apt,ro \
    --mount=type=bind,source=docker/prepare_runtime_rootfs.sh,target=/usr/local/bin/prepare-runtime-rootfs,ro \
    /usr/local/bin/snapshot-apt "${UBUNTU_SNAPSHOT}" python3.12 ca-certificates \
    && /usr/local/bin/prepare-runtime-rootfs

FROM cpu-rootfs AS cpu
ARG CHECKMAITE_VERSION=0.0.0
LABEL org.opencontainers.image.title="CheckMAITE batch container" \
      org.opencontainers.image.description="Finite single-host CheckMAITE execution (CPU, Ubuntu 24.04)" \
      org.opencontainers.image.version="${CHECKMAITE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite"
COPY --from=cpu-environment --chown=0:0 /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHECKMAITE_OUTPUT_DIR=/output/results \
    CHECKMAITE_CACHE_DIR=/cache
USER 10001:10001
WORKDIR /checkmaite
STOPSIGNAL SIGTERM
HEALTHCHECK NONE
ENTRYPOINT ["/opt/venv/bin/checkmaite-container"]
CMD ["run"]

# CUDA remains derived from NVIDIA's pinned runtime image so its complete ENV
# and compatibility matrix are inherited rather than copied into this file.
FROM cuda-rootfs AS cuda
ARG CHECKMAITE_VERSION=0.0.0
LABEL org.opencontainers.image.title="CheckMAITE batch container" \
      org.opencontainers.image.description="Finite single-host CheckMAITE execution (NVIDIA CUDA, Ubuntu 24.04)" \
      org.opencontainers.image.version="${CHECKMAITE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-NVIDIA-Proprietary" \
      org.opencontainers.image.source="https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite"
COPY --from=cuda-environment --chown=0:0 /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHECKMAITE_OUTPUT_DIR=/output/results \
    CHECKMAITE_CACHE_DIR=/cache
USER 10001:10001
WORKDIR /checkmaite
STOPSIGNAL SIGTERM
HEALTHCHECK NONE
ENTRYPOINT ["/opt/venv/bin/checkmaite-container"]
CMD ["run"]

FROM cpu AS final
