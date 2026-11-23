# Furiosa compiler

A Triton backend (`TRITON_PLUGIN_DIRS` plugin, built into libtriton) plus a driver that turns a pure-Triton
model file into a furiosa-opt crate. Pipeline:

```
model.py ── driver: one triton.compile per launch specialization, every integer scalar a constexpr
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
The compiler venv also needs CPU-only torch for `kernel[grid](...)` on host tensors:
`uv pip install --python compiler/.venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu`.

## Triton launches run on the NPU

In the compiler venv the furiosa driver is always Triton's active driver (`compiler/backend/driver.py`), so
calling a `@triton.jit` kernel, `kernel[grid](x, y, out, n, BLOCK=1024)`, runs it on npu3: Triton compiles its
TTIR, then the launcher specializes the launch (integer scalars as constexprs, float ones as fp32 values the raise
binds, the grid, each tensor's dtype and shape), compiles it through every stage into a crate, builds it once per
specialization under `~/.cache/furiosa-npu`, runs it on npu3 and copies the written tensors back
(`furiosa_compiler/npu.py`).
Nothing falls back to the CPU or GPU: a kernel the compiler cannot map raises its diagnostic.
`compiler/tests/test_npu_launch.py` checks it.

## Layout

| Path | Owns |
|---|---|
| `backend/` | the Triton backend: options, stages |
| `furiosa_compiler/` | the driver (`python -m furiosa_compiler`) |
| `lib/Transforms/` | raising: TTIR → whole-tensor linalg |
| `lib/Runtime/` | L1: JIT-compiling raised linalg on the CPU |
| `lib/Codegen/` | the furiosa-opt dialect (fopt), linalg → fopt mapping, fopt → Rust |
| `bindings/` | one nanobind file per area, called from `triton_furiosa.cc` |
| `tools/` | `furiosa-mlir-opt`: mlir-opt with the Furiosa passes and dialects, for the lit tests |
| `tests/lit/` | per-pass tests: one small input, one or more passes, FileCheck on the output |

## Per-pass tests

```
compiler/build.sh                        # builds libtriton and furiosa-mlir-opt
compiler/.venv/bin/lit compiler/tests/lit -v
```

Each test names one contract of one pass in a comment (a raise rule, a structure promotion, a diagnostic)
and checks it with FileCheck; absences are checked over the whole output with `--implicit-check-not`.
Inputs are TTIR of small Triton kernels with their `furiosa.launch` attribute, the kernel source quoted on
top. A change to a pass comes with a test for the behavior it adds or fixes; the harness runs these before
the cases (`unit:` line) and fails the run on a broken one.

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
- Iteration spaces follow the buffers: a store's generic iterates a view of the stored buffer whose dims
  are the buffer's own dims where the tiles allow (a flattened index tiled across dims, such as a pixel
  index over `[N, OH, OW]`, becomes those dims). A tile index that the computation uses on its own (a
  head index within a `[T, H*D]` row) gets a dim of its own. Floor-divisions and remainders that the
  dims' ranges decide are resolved.
- Generics and kept loop states span only the dims they depend on, and a reduction over a padded tile
  (`BLOCK_K > K`, masked to zero) runs over the live extent.
- Function arguments keep the Triton parameter names as `NameLoc`s.

Running the function on the buffers must give exactly what the Triton grid gives (judged by harness L1).

## Contract: driver output

`python -m furiosa_compiler <model.py> -o <dir>` writes:

```
<dir>/graph.json                     every graph: buffers and launches (kernel symbol, buffer per pointer arg, grid)
<dir>/kernels/<symbol>/{ttir,linalg,structured,steps,tiled}.mlir
```

`structured.mlir` is the raised kernel after `furiosa-structure` (affine reads as indexing-map inputs).
`steps.mlir` is what codegen consumes (lib/Codegen/Steps.cpp, Hoist.cpp): the structured kernel with every
index-only value evaluated into a constant table, and, where no op class maps the kernel whole, split into
steps: private functions in the raised-kernel contract that one op class maps each, called in order by the
kernel on its buffers and `tensor.empty` temporaries. The crate launches the steps in call order and uploads
the tables.
`tiled.mlir` is `steps.mlir` after `furiosa-tile` (lib/Codegen/Tile/TilePass.cpp), the first level of codegen's
progressive lowering: each kernel gets the loops its op class's tiling hook asks for (`OpClass::tiling`, a
class-neutral `mapping::Tiling`; the class maps the tiled kernel, reading the loops back with
Tile/TileLoops.h). The tiled class tiles by its plan (Tile/Plan.h, the cost model holding the NPU's capacities:
TRF lanes, the DM budget, intra-slice accumulators, slices, lanes) into
`scf.for` loops over its output rows (upstream TilingInterface tile and fuse; pads and buffers stay whole and
are sliced) and K loops over its contraction's reduction dims (`furiosa.tile_dim` names the dim; a K tile is
whole taps or part of the inner axis), and mapping to fopt folds the loops back into the plan's tile digits and
K tiles. Codegen maps this stage. L1
runs `linalg.mlir`, `structured.mlir`, `steps.mlir` and `tiled.mlir`.
Mapping a tiled kernel to fopt is the second level: the tiled class emits its contraction stages as fopt, its
Vector Engine work as one abstract op per operation (`fopt.ve`: the ALU call, its operand, or a zip of two tiles), its tiles as
device loops (one `fopt.for` per tile digit, the body once with its tile views indexed by the loop's induction
variable; furiosa-opt keeps the loop), and its data movement abstract, as `fopt.stage` (a buffer read at a linear index over the program's axes, with the layout its
tiles are cut from) and `fopt.read` (one tile of it in DM). `furiosa-dma` (lib/Codegen/Tile/Dma.cpp) makes the
data-movement decisions and lowers both to HBM staging buffers and DMAs: padded copies and their fills (one copy
shared by the stages of a buffer that differ only in their windows, such as unrolled taps), tap-major or
row-major K for contraction operands, inner digits zero-padded in the source copy (a K window of a longer dim,
such as attention's P x V reading one key block of V, is one block of the padded K digit, the dim padded to whole blocks in the copy when they do not divide it), the taps' DMAs as kept loops
over affine tap classes. `furiosa-ve-fuse` (lib/Codegen/Tile/VeFuse.cpp) groups the `fopt.ve` ops into passes:
an op joins the pass of its input when that input has no other use and the op fits the engine's stage order
(stages never go back; the float and clip stages take several ops, each on its own ALU; multiplications take
Mul0, Mul1, then the FMA unit; an add moves to the clip stage once the FMA unit is taken), so a batch norm's
subtract, scale and shift share one fetch and one commit. The crate writes each kernel's level before both passes
as `<stem>.ftile.mlir`, next to the final `<stem>.fopt.mlir`.
fp16 buffers live as f32 on the device (furiosa-opt has no f16): mapping a kernel to fopt widens its `f16` to
`f32` and drops the conversions between them, and the crate converts F16 at the graph boundary (exactly in,
round-to-nearest-even out); `steps.mlir` and the stages before it keep `f16`.

For N chips (`FURIOSA_CHIPS`, docs/usage.md) every kernel's chip mapping is the `CHIP` axis, and `furiosa-shard` runs
first: it splits the outermost parallel dim of a result whose blocks are boxes of one dim of every operand (traced
through the linalg ops in both directions) and whose block maps (the whole mapping, tried on a copy), and marks the
split arguments and results `furiosa.shard`. The class maps the block; its kernel takes and returns the whole
buffers (chip mapping `m![N]`, the same on every chip), copies the chip's block of each split argument out of them
and gathers the result's blocks into every chip's copy (`KernelBuilder`; lit: mapping/chips.mlir).

A specialization is keyed by kernel, constexpr and float scalar values, grid and the dtype/shape of each pointer
argument (not the buffer names), so one compiled kernel serves every launch with that shape; `graph.json`
binds buffers per launch:

```json
{"kernels": {"add_kernel_0": {"kernel": "add_kernel", "pointer_args": ["x_ptr", "y_ptr", "out_ptr"],
                              "written": [2], "linalg": "kernels/add_kernel_0/linalg.mlir"}},
 "graphs": [{"name": "...", "buffers": {"x": {"kind": "input", "dtype": "fp32", "shape": [2048]}},
             "launches": [{"kernel": "add_kernel_0", "args": ["x", "y", "out"], "grid": [4, 1, 1]}]}]}
```
