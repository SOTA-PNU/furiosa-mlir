# The furiosa-mlir playground: the compiler (from the binary package), the Rust toolchain furiosa-opt builds
# with (cargo registry and dependencies warmed), JupyterLab, the example models and the playground.
#
#   scripts/package.sh
#   docker build -f docker/Dockerfile --build-arg PACKAGE="$(ls dist/furiosa-mlir-*.tar.gz)" -t furiosa-mlir .
#   docker/run.sh                    # JupyterLab on :8888, npu3 only;  docker/run.sh bash  for a shell
FROM ubuntu:24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv build-essential clang-18 libclang-18-dev curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# The toolchain the generated crates pin (rust-toolchain.toml).
ENV RUSTUP_HOME=/opt/rust CARGO_HOME=/opt/rust PATH=/opt/rust/bin:$PATH LIBCLANG_PATH=/usr/lib/llvm-18/lib
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal \
        --default-toolchain nightly-2026-05-01 --component rustfmt --component clippy

ARG PACKAGE
ADD ${PACKAGE} /opt/
RUN ln -s /opt/furiosa-mlir-*-linux-x86_64 /opt/furiosa-mlir && /opt/furiosa-mlir/install.sh \
    && ln -s /opt/furiosa-mlir/bin/* /usr/local/bin/ \
    && /opt/furiosa-mlir/venv/bin/pip install --quiet jupyterlab \
    && ln -s /opt/furiosa-mlir/venv/bin/jupyter /usr/local/bin/jupyter

# One build slot (a single user), warmed: a model's crate built once fetches and compiles every dependency.
# Building needs no NPU; running does (docker/run.sh passes npu3 in).
ENV FURIOSA_VISIBLE_CHIPS=3 FURIOSA_BUILD_SLOTS=1
RUN furiosa compile /opt/furiosa-mlir/examples/models/lenet5.py -o /tmp/warm \
    && /opt/furiosa-mlir/venv/bin/python -c "from pathlib import Path; from furiosa_compiler.npu import build_run_graph; build_run_graph(Path('/tmp/warm/crate'), Path('/tmp/warm/run_graph'))" \
    && rm -rf /tmp/warm

WORKDIR /workspace
RUN cp -a /opt/furiosa-mlir/examples/playground /opt/furiosa-mlir/examples/models /workspace/
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.root_dir=/workspace"]
