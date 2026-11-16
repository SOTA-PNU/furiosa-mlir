# Furiosa compiler

A Triton backend (`TRITON_PLUGIN_DIRS` plugin, built into libtriton) plus a driver that turns a pure-Triton
model file into a furiosa-opt crate. Pipeline:

```
model.py ── driver: one triton.compile per launch specialization, every scalar a constexpr
         ── backend stages: ttir ─▶ linalg (raise) ─▶ fopt (mapping) ─▶ rust
         ── driver: crate = kernels + run_graph host program from the launch graph
```

## Build

```
uv venv compiler/.venv --python 3.12
uv pip install --python compiler/.venv/bin/python "setuptools>=40.8.0" "cmake>=3.20,<4.0" "ninja>=1.11.1" "nanobind==2.10.2" wheel lit numpy ml_dtypes
TRITON_PLUGIN_DIRS=$PWD/compiler TRITON_BUILD_DIR=$PWD/build/triton TRITON_BUILD_PROTON=OFF MAX_JOBS=48 \
  uv pip install --python compiler/.venv/bin/python --no-build-isolation -e third_party/triton
compiler/build.sh          # incremental rebuild of libtriton after C++ changes (serialized by a lock)
```

`triton/backends/furiosa` is a symlink to `compiler/backend`, so Python changes need no rebuild.

## Layout

| Path | Owns |
|---|---|
| `backend/` | the Triton backend: options, stages |
| `furiosa_compiler/` | the driver (`python -m furiosa_compiler`) |
| `lib/Transforms/` | raising: TTIR → whole-tensor linalg |
| `lib/Runtime/` | L1: JIT-compiling raised linalg on the CPU |
| `lib/Codegen/` | the furiosa-opt dialect (fopt), linalg → fopt mapping, fopt → Rust |
| `bindings/` | one nanobind file per area, called from `triton_furiosa.cc` |

## Contract: raised linalg (stage `linalg`)

One `builtin.module` per kernel specialization holding one public `func.func @<kernel name>`:

- One argument per pointer argument of the Triton kernel, in kernel order, typed as the whole graph
  buffer: `tensor<SHAPE x T>` with the buffer's declared shape and dtype (`fp32`→`f32`, `bf16`, `fp16`→`f16`,
  `i32`, `i64`).
- One result per pointer argument the kernel stores to, in kernel order: the whole buffer after the
  kernel. Elements the kernel does not write keep the argument's values.
- `furiosa.written = [i, ...]` on the function: the pointer-argument indices of its results.
- The body is the kernel's effect on whole tensors for the entire grid: linalg ops (`linalg.generic` and
  named ops), `tensor` reshapes/slices/pads, `arith`/`math` in bodies, `scf.for` only where a loop carries
  state that is not a plain reduction. No `tt` ops remain.
- Raised form is gather form: a generic's body reads buffers with `tensor.extract` at addresses computed
  from `linalg.index` (masked reads are selects over clamped addresses; data-dependent addresses are just
  more arithmetic). A generic with reduction dims carries their extents on a `tensor.empty` input it never
  reads. A later generic pass (`furiosa-structure`) promotes affine extracts to indexing-map inputs, so the
  mapping stage sees ordinary structured linalg.

Running the function on the buffers must give exactly what the Triton grid gives (judged by harness L1).

## Contract: driver output

`python -m furiosa_compiler <model.py> -o <dir>` writes:

```
<dir>/graph.json                     every graph: buffers and launches (kernel symbol, buffer per pointer arg, grid)
<dir>/kernels/<symbol>/{ttir,linalg}.mlir
```

A specialization is keyed by kernel, constexpr values, grid and the dtype/shape of each pointer
argument (not the buffer names), so one compiled kernel serves every launch with that shape; `graph.json`
binds buffers per launch:

```json
{"kernels": {"add_kernel_0": {"kernel": "add_kernel", "pointer_args": ["x_ptr", "y_ptr", "out_ptr"],
                              "written": [2], "linalg": "kernels/add_kernel_0/linalg.mlir"}},
 "graphs": [{"name": "...", "buffers": {"x": {"kind": "input", "dtype": "fp32", "shape": [2048]}},
             "launches": [{"kernel": "add_kernel_0", "args": ["x", "y", "out"], "grid": [4, 1, 1]}]}]}
```
