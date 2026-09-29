# llamAmpere sm_86 (RTX 3090) — VAST.ai template payload (org-xpi U2)
# Base matches uncanny's CUDA 12.4 (nvcc V12.4.131, driver 550.54.15).
# Builds llamAmpere IN-IMAGE pinned to 31cdb8e22 (release/v0.3); the 14.5 GiB
# ATX GGUF is NOT in the image — entrypoint.sh downloads + verifies it at boot.
#
# Build flags mirror uncanny's as-deployed build-sm86 CMakeCache (parity gate,
# org-xpi.2): Release, GGML_CUDA=ON, GGML_CUDA_FA=ON, archs=86,
# toolchain /usr/local/cuda-12.4. GGML_NATIVE is intentionally OFF so the image
# boots on *any* sm_86 box, not just the CPU it was compiled on.
#
# Stage 1: compile llama-server. Stage 2: slim runtime (binary + assets only).
ARG BASE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG BASE_RT=nvidia/cuda:12.4.1-base-ubuntu22.04

############################## build stage ##############################
FROM ${BASE} AS builder
ARG LLAMAAMPRE_COMMIT=31cdb8e2257aeb855462d1697e68b14738818f49
# Cap parallel jobs: CUDA kernel builds are RAM-hungry (~1-2 GB per nvcc).
# Default to a conservative 4; raise on beefier build boxes.
ARG BUILD_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/root/.ccache BUILD_JOBS=${BUILD_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential ccache cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/JakeATX/llamAmpere.git llamAmpere \
    && cd llamAmpere \
    && git checkout --detach ${LLAMAAMPRE_COMMIT} \
    && git rev-parse HEAD \
    # --- driver-link fix (org-xpi U2) -------------------------------------
    # This base image ships the CUDA driver STUB (/usr/local/cuda/lib64/stubs/
    # libcuda.so, SONAME libcuda.so.1) but no real driver. Two problems:
    #  1. FindCUDAToolkit (cmake 3.22 in this image) resolves CUDA::cuda_driver
    #     to an EMPTY IMPORTED_LOCATION, so ggml-cuda never gets the link arg.
    #  2. Even when linked, the FINAL llama-server executable link can't find
    #     the stub's SONAME libcuda.so.1 (it is named libcuda.so), leaving
    #     cu* symbols (cuMemCreate, cuDeviceGet, ...) unresolved.
    # Fix: publish the stub under its SONAME (libcuda.so.1) in the stubs dir,
    # patch ggml-cuda to link the stub by path, and pass -Wl,-rpath-link at
    # configure time so the FINAL llama-server exe link resolves the transitive
    # libcuda.so.1. Build stage only — the runtime stage must NOT ship the
    # stub, or it could shadow the real driver injected by the NVIDIA
    # container runtime on VAST / docker --gpus.
    && ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && sed -i 's#CUDA::cuda_driver#/usr/local/cuda/lib64/stubs/libcuda.so#g' \
        ggml/src/ggml-cuda/CMakeLists.txt \
    && grep -n 'stubs/libcuda.so' ggml/src/ggml-cuda/CMakeLists.txt

WORKDIR /build
RUN ccache -M 2G; \
    cmake -S /src/llamAmpere -B /build -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_FA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_CUDA_TOOLCHAIN_ROOT=/usr/local/cuda-12.4 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
      -DGGML_NATIVE=OFF \
      -DGGML_CCACHE=ON \
      -DGGML_CUDA_GRAPHS=ON \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DGGML_BUILD_TESTS=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build /build --target llama-server -j"${BUILD_JOBS}"

############################## runtime stage ##############################
# Runtime base is the CUDA 12.4.1 BASE image (244 MB, no -devel, no cudnn)
# vs the 9.3 GB devel base used by the builder. llamAmpere needs cudart (in
# base), cublas (copied from builder stage below), NCCL (pip wheel below),
# libgomp (apt below) and the driver (injected by the NVIDIA container
# runtime at boot). cuDNN / nvcc / headers are build-only.
FROM ${BASE_RT} AS runtime
ARG LLAMAAMPRE_COMMIT=31cdb8e2257aeb855462d1697e68b14738818f49
ENV DEBIAN_FRONTEND=noninteractive
ENV LLAMAAMPRE_COMMIT=${LLAMAAMPRE_COMMIT}
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.revision="${LLAMAAMPRE_COMMIT}"
LABEL org.opencontainers.image.source="https://github.com/alx/llamampere-sm86"

# llamAmpere links nvidia-nccl (as on uncanny: pip nvidia-nccl lib).
# Pull the nccl shared libs from the pip wheel into /opt/nccl, then drop pip.
# libgomp: ggml-cpu/ggml-base link libgomp, not in CUDA base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip curl ca-certificates tini libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir "nvidia-nccl-cu12" \
    && NCCL_LIB=$(python3 -c "import nvidia.nccl as n, os;print(os.path.join(n.__path__[0],'lib'))") \
    && install -d /opt/nccl/lib \
    && cp -L "$NCCL_LIB"/libnccl.so* /opt/nccl/lib/ \
    && python3 -m pip uninstall -y nvidia-nccl-cu12 \
    && rm -rf /root/.cache /home/app/.cache \
    && ldconfig

RUN useradd -m app \
    && install -d -o app -g app /opt/llama/bin /opt/llama/templates /opt/llama/vocab /models

# llamAmpere binary + ALL shared libs it links (org-xpi.7: the fork builds with
# BUILD_SHARED_LIBS=ON on Linux, so llama-server is a thin exe needing
# libllama-server-impl.so etc.). CMAKE_LIBRARY_OUTPUT_DIRECTORY puts every .so
# next to the exe in /build/bin. COPY the whole directory (directory COPY
# preserves the .so -> .so.N symlinks; a per-file glob would dereference and
# triple the 260 MB libggml-cuda). LD_LIBRARY_PATH below includes this dir.
COPY --from=builder /build/bin/ /opt/llama/bin/
# cublas + cublasLt runtime — in the devel builder base, absent from -base.
# libggml-cuda DT_NEEDEDs both libcublas.so.12 and libcublasLt.so.12. Copy the
# two REAL versioned files by exact name (BuildKit COPY globs from another
# stage are unreliable / cache-sticky); then recreate the soname symlinks.
# Version 12.4.5.8 is pinned to the cuda 12.4.1 devel base image.
COPY --from=builder /usr/local/cuda-12.4/lib64/libcublas.so.12.4.5.8    /usr/local/cuda-12.4/lib64/
COPY --from=builder /usr/local/cuda-12.4/lib64/libcublasLt.so.12.4.5.8 /usr/local/cuda-12.4/lib64/
RUN set -eux; cd /usr/local/cuda-12.4/lib64; \
    ln -sf libcublas.so.12.4.5.8    libcublas.so.12; \
    ln -sf libcublasLt.so.12.4.5.8  libcublasLt.so.12; ldconfig

# P2 runtime assets — pulled from uncanny's as-deployed copies (org-xpi U2).
# These are the exact files the production llama-server uses.
COPY assets/Qwen3.8-Sharp.jinja /opt/llama/templates/Qwen3.8-Sharp.jinja
COPY assets/atx_65536.txt       /opt/llama/vocab/atx_65536.txt
COPY entrypoint.sh              /opt/llama/entrypoint.sh
RUN chown -R app:app /opt/llama /models \
    && chmod +x /opt/llama/entrypoint.sh \
    && ldconfig

# /opt/llama/bin: app .so's next to the exe; then NCCL; then CUDA runtime
# (cudart, cublas). Driver libcuda comes from the nvidia runtime injection.
ENV LD_LIBRARY_PATH=/opt/llama/bin:/opt/nccl/lib:/usr/local/cuda-12.4/lib64 \
    GGML_Q8_TURBO3_MMA_FUSED=1

USER app
WORKDIR /models
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/llama/entrypoint.sh"]
