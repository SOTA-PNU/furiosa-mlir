# Using furiosa-mlir

Triton compiled for, and run on, the Furiosa RNGD NPU (`npu3`). There are three ways in: a source checkout,
the binary package, and the Docker playground. All three give the same `furiosa` CLI and the same
`kernel[grid](...)`.

## Install

| | How | CLI |
|---|---|---|
| Release binaries | download from the GitHub release (`furiosa-mlir-<version>-linux-x86_64.tar.gz` or the wheel) | as below |
| Binary package | untar, run its `install.sh` | `<package>/bin/furiosa` |
| Wheel | `pip install furiosa_mlir-<version>-cp312-cp312-linux_x86_64.whl` into a fresh Python 3.12 venv | `furiosa` |
| Source | `compiler/build.sh` (compiler/README.md) | `compiler/furiosa` |
| Docker | the package, then `docker build` (below) | `furiosa` in the container |

```
gh release download v1.3.0 -R SOTA-PNU/furiosa-mlir -p '*.tar.gz'      # or -p '*.whl'
tar xzf furiosa-mlir-1.3.0-linux-x86_64.tar.gz && furiosa-mlir-1.3.0-linux-x86_64/install.sh

python3.12 -m venv venv && venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    furiosa_mlir-1.3.0-cp312-cp312-linux_x86_64.whl          # the wheel provides `triton`: use a fresh venv
cargo +nightly-2026-05-01 binstall cargo-furiosa-opt@0.10.0  # or cargo install; on PATH for NPU builds
```

Both hold the furiosa-enabled Triton (libtriton with the compiler passes, stripped), `furiosa_compiler`,
`furiosa-mlir-opt` and the licenses of Triton and LLVM; the tarball adds the example models and the playground,
and its `install.sh` makes a Python 3.12 venv beside it with torch (CPU) and installs the furiosa-opt tools
(`cargo-furiosa-opt` 0.10.0, crates.io) into `.tools/`. Running on the NPU needs Rust (rustup): the generated
crates pin their toolchain, and rustup installs it on the first build. `scripts/package.sh` builds both artifacts
from a built tree into `dist/`.

## A kernel

In the compiler's Python, a launch runs on the NPU:

```python
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)   # CPU tensors in, results written back
```

The first launch of a specialization (kernel, constexprs, grid, dtypes and shapes) compiles it and builds its
crate, which takes about a minute; later launches reuse the build. A kernel the compiler cannot map raises its
diagnostic. Examples: `playground/vector_add.py`, `playground/compile_only.py` (stages without running), and
`playground/playground.ipynb`.

## A model: compile, run, trace

A model file defines `graphs()`, its `@triton.jit` launches in order, and the buffers they use (`models/*.py`).

```
furiosa compile models/lenet5.py -o out/lenet5
furiosa run out/lenet5 [--graph 0] [--inputs in.safetensors] [--outputs out.safetensors] [--emulation]
furiosa trace models/lenet5.py -o out/lenet5-trace --npu --dump out/lenet5-trace/dump
```

- **compile** writes every kernel through every stage (`out/compiler/kernels/<kernel>/`, table below),
  `out/compiler/graph.json` (the launches and buffers), and the furiosa-opt crate (`out/crate/`).
- **run** builds the crate and runs one graph on npu3, or with `--emulation` on furiosa-opt's CPU emulation.
  Inputs come from a safetensors file keyed by buffer name (the graph's `input`, `weight` and `state`
  buffers). Without `--inputs`, random ones are written to `out/inputs-<graph>.safetensors`, so a run can be
  repeated or compared. It prints each output's shape and range.
- **trace** runs the graph launch by launch on the CPU reference, the raised kernels on the LLVM JIT. With
  `--npu`, each launch also runs on npu3 from the reference's buffers, so one launch's error does not feed
  the next. It prints every buffer each launch writes, with its largest value and the NPU's error, and marks the
  launches past `--rtol`/`--atol`. `--dump` saves each launch's buffers: `NNN_<kernel>.safetensors` (reference),
  `NNN_<kernel>.npu.safetensors` (npu3), and `inputs.safetensors`.

```
  #  kernel                       buffer                 max|ref|  npu max err       rel
  0  conv2d_tanh_kernel_0         conv1.out                0.9984     0.006909    0.0069
  1  avgpool2d_kernel_0           pool1.out                 0.976            0         0
```

## Several chips

`FURIOSA_CHIPS=N` compiles every kernel for N chips (SPMD) and runs it on the N NPUs `FURIOSA_VISIBLE_CHIPS` names;
it is part of every compile (and of the launch cache key), and N = 1 is the single-chip compiler unchanged.

```
FURIOSA_CHIPS=2 furiosa compile models/vgg16.py -o out/vgg16-2      # kernels for two chips
FURIOSA_CHIPS=2 furiosa run out/vgg16-2                            # on npu2 and npu3
```

Between kernels every buffer is whole on every chip: uploads write each chip's copy (in DRAM the other chips may
write), downloads read chip 0's. Every kernel runs on every chip (every chip mapping is the `CHIP` axis), and each
kernel (each step of a split kernel) splits the outermost parallel dim of a result whose blocks are boxes of one dim
of every operand (a matmul's rows, an image's output channels, a row-wise op's rows) and whose block still maps
(`furiosa-shard`, lib/Codegen/LinalgToFopt.cpp): chip c copies block c of the inputs that dim indexes out of its
own copy, computes block c of the result, and the result's blocks are gathered into every chip's copy (an
all-gather: a DMA across chips). The other inputs are read whole. A slice that keeps the dim whole (a pool's window
reads, split by channel) or a pad of other dims is a block of its source's block. A kernel with no such dim (one whose
blocks would be windows, use the dim's index as data, or do not divide by N), or that reads a buffer it writes, runs
whole on every chip. `FURIOSA_DEBUG=1` says why a dim does not split.

The NPUs are shared: npu0-npu2 run other users' services. A run opens exactly the NPUs it uses, holds a lock per
NPU (`~/.cache/furiosa-npu/npu.lock` for npu3, `npu<i>.lock` for the others) and, right before it opens them, waits
until every PE of each is idle (`/sys/class/rngd_mgmt/rngd!npu<i>mgmt/pe_occupancy`), up to 30 minutes; it never
preempts another user's run. Single-chip work stays on npu3; take the other NPUs only for N-chip runs, while idle,
briefly.

## Debugging the stages

Each kernel's stages, in pipeline order (compiler/README.md describes the passes):

| File | Stage |
|---|---|
| `ttir.mlir` | Triton IR from the frontend |
| `linalg.mlir` | raised to whole-tensor linalg (`furiosa-raise-to-linalg`) |
| `structured.mlir` | reads promoted to indexing maps (`furiosa-structure`) |
| `steps.mlir` | split into steps an op class can map |
| `tiled.mlir`, `ftile.mlir` | tiled into loops (`furiosa-tile`), then mapped to tile-level fopt |
| `fopt.mlir` | furiosa-opt ops, with data movement (`furiosa-dma`) and fused Vector Engine passes |
| `kernel.rs` | the kernel's Rust in the crate |

- **Replay a pass:** `furiosa-mlir-opt --furiosa-structure linalg.mlir`. The per-pass lit tests in
  `compiler/tests/lit` show every pass's flags. The raising pass reads the launch from the module attribute
  `furiosa.launch`, which `linalg.mlir` carries.
- **See the IR before every pass:** `MLIR_ENABLE_DUMP=1` (stderr), with `TRITON_ALWAYS_COMPILE=1` to skip
  Triton's cache.
- **Why a kernel does not map:** `FURIOSA_DEBUG=1` prints why each op class or candidate step passes, and the
  raised function when it fails to verify. `FURIOSA_TILED_DEBUG=1` prints the tiler's trial mappings.
- **A launch that is wrong on the NPU:** `furiosa trace --npu` names it. Its stages are in `out/compiler/kernels/`,
  and `furiosa-mlir-opt` replays them.
- **Against a GPU golden:** the harness (harness/README.md) checks every launch and graph from L1 (the raised IR
  on the CPU) to L4 (npu3).

| Variable | Default | |
|---|---|---|
| `FURIOSA_CHIPS` | `1` | the chips every kernel runs on (below) |
| `FURIOSA_VISIBLE_CHIPS` | the last `FURIOSA_CHIPS` NPUs (`3`; `2,3`; ...) | the NPUs the runs open, `FURIOSA_CHIPS` of them |
| `FURIOSA_NPU_CACHE` | `~/.cache/furiosa-npu` | kernel builds and cargo target dirs |
| `FURIOSA_BUILD_SLOTS` | cores / 16 | cargo builds at once |
| `FURIOSA_RUNNER_UTILS` | beside libtriton, else the build's | the MLIR runner utilities the JIT links |

## Docker playground

```
scripts/package.sh
docker build -f docker/Dockerfile --build-arg PACKAGE="$(ls dist/furiosa-mlir-*.tar.gz)" -t furiosa-mlir .
docker/run.sh                 # JupyterLab on http://localhost:8888 (PORT=...), npu3 only
docker/run.sh bash            # a shell with the furiosa CLI
```

The image has the compiler, the Rust toolchain with the cargo registry and the crates' dependencies already
built, and JupyterLab. `/workspace` holds the playground and the models. `docker/run.sh` passes in npu3's device
nodes and nothing else, and mounts `WORKSPACE` (default `./playground`) at `/workspace/mine`.
