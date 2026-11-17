# Compiler harness

Defines what "correct" means for the Triton → furiosa-opt compiler before the compiler exists, and
measures it level by level. Every level judges a file the previous level produced, so each runs on
its own and a failure points at one stage.

## Cases

A case is a pure-Triton file with the same shape as a model file in `models/`: the launch-graph block
(`Buf` / `Launch` / `Graph`, copied verbatim from `tests/fixtures/tiny_mlp.py`), kernels, and
`graphs()`. Micro cases live in `harness/cases/<name>.py`; the model files are cases too.
`harness/registry.py` lists every case with its tolerances and input ranges.

## Golden data

`python -m harness.golden <case>` runs the case's graphs on the GPU through `tests/harness.py` (the
Triton JIT is the reference) and writes, per graph index `i`:

```
harness/golden/<case>/<i>/inputs.safetensors    every input, weight and state buffer before graph i
harness/golden/<case>/<i>/outputs.safetensors   every output and state buffer after graph i
harness/golden/<case>/meta.json                 graph names, buffers, Triton version, seed
```

Tensors are stored under their graph buffer names, in the dtype and row-major shape the graph
declares. Graphs run in order on one set of buffers, so state (e.g. a KV cache) carries from graph
`i` into graph `i + 1`. Inputs are seeded random; `registry.py` can bound integer inputs (token ids).

## The crate contract

A crate under test is a furiosa-opt kernel package — hand-written (`harness/reference/<case>/`) now,
compiler-generated later — that has a binary named `run_graph`:

```
run_graph <graph-index> <inputs.safetensors> <outputs.safetensors>
```

It reads every input, weight and state buffer of that graph by name from the inputs file, runs the
graph, and writes every output and state buffer by name to the outputs file, in the graph's dtype
and row-major shape. Whatever device layout or padding the kernels use is the crate's business; the
files are always the canonical host layout.

## Levels

| Level | Artifact | Pass when | Needs |
|---|---|---|---|
| L0 `ttir` | `harness/out/<case>/<case>.triton.mlir` | `scripts/dump_mlir.py` succeeds | triton |
| L1 `raised` | raised `linalg` module | it runs on MLIR's CPU path and matches golden (exact dtype) | compiler (not yet) |
| L2a `codegen` | the crate | the compiler mapped every kernel and wrote the crate (no `CODEGEN_FAILED`); seconds, no build | compiler |
| L2 `compile` | the crate | `cargo furiosa-opt build --bin run_graph` succeeds: furiosa-opt accepts every mapping | furiosa-opt |
| L3 `cpu` | the crate | plain `cargo run --bin run_graph` (furiosa-opt's CPU backend) matches golden | furiosa-opt |
| L4 `npu` | the crate | the NPU build on npu3 matches golden | an idle npu3 |

L1 is two checks, both required:

- **Per launch** (teacher forcing): golden records, per launch, the buffers it binds after it ran
  (`harness/golden/<case>/<i>/launches/<j>.safetensors`). Each raised kernel runs on the golden state just
  before its launch, and the buffers it writes must match the golden state just after it, within the
  case's `exact` tolerance raised to one unit in the last place of the dtype (bf16 2^-7, fp16 2^-10,
  relative to max|golden|): the kernel's arithmetic is reproduced in its own dtypes, but Triton leaves
  the order of fp32 accumulation unspecified, so a final rounding may differ by one ulp. Errors cannot
  compound, so a failure names the wrong kernel.
- **Whole graph**: the launches chained from the graph's inputs, compared with the case's `graph`
  tolerance (default: `exact`). Only deep reduced-precision models set it, with the reason in
  `registry.py`, because one-ulp differences compound over their depth.

For the fp32 models, the NPU tolerance is measured, not chosen: twice the logit drift of the same
Triton graph run entirely in bf16 on the GPU (recorded in `registry.py`), plus the top-1 class must be
unchanged (`top1`). Qwen3 is a bf16 graph: its NPU tolerance is twice the drift between two correct implementations of
the same graph (L1's CPU JIT against the GPU golden, 2.7%), and its greedy `next_token` must match
exactly (integers are always exact).

The Vector Engine's own arithmetic is measured, not assumed: `harness/characterize/` is a furiosa-opt crate that
runs every f32 unary function of the Vector Engine over fixed input ranges on npu3 and compares with an f64
reference; `npu3.jsonl` records the result. Measured: exp, log, sqrt, tanh, sigmoid, erf, sin and cos are within
1.7 f32 ulps, but sin and cos return non-finite values for |x| > 2π, so a kernel's sin/cos must reach the
hardware range-reduced (the compiler may not drop a kernel's own reduction, and must add one where the input
range is not known to fit). Rerun after a firmware or furiosa-opt change:

```
cd harness/characterize && cargo furiosa-opt build --release --bin characterize   # CARGO_TARGET_DIR=../out/target/npu
FURIOSA_VISIBLE_CHIPS=3 ../out/target/npu/release/characterize > npu3.jsonl
```

L3 and L4 compare with the case's NPU tolerance, which allows for the dtype policy the NPU forces
(no fp32 or fp16 contraction: dots run in bf16 with fp32 accumulation; fp16 buffers live as f32 on the
device, and F16 is converted at the graph boundary). A match means NaN exactly where golden has NaN
(a function outside its domain) and `max|out - golden| <= atol + rtol * max|golden|` over the other
elements, for every output; integers match exactly.

## The compiler under test

`--impl compiler` runs the compiler (in `compiler/.venv`, see `compiler/README.md`) once per case:

```
python -m furiosa_compiler <case.py> -o harness/out/<case>/compiler --crate harness/out/<case>/crate
python -m furiosa_compiler.l1_launch harness/out/<case>/compiler harness/golden/<case> --rtol R --atol A
python -m furiosa_compiler.l1 harness/out/<case>/compiler harness/golden/<case> --rtol R --atol A
```

The first writes the per-kernel stages, `graph.json` and the crate; the second executes every graph's
launches on the CPU with the raised kernels (L1) and prints one JSON line,
`{"status": "PASS" | "FAIL", "detail": "..."}`.

## NPU-only rule

Triton code compiled by this compiler always lowers to the Furiosa NPU and runs there; the harness
fails the run on any way around that:

- `npu-only:` before the cases, it scans the compiler's product sources (not `compiler/tests/`) for
  another Triton target (`GPUTarget("cuda"|"hip")`), another driver (`TRITON_DEFAULT_BACKEND` set to
  anything but `furiosa`, `driver.set_active`), or CUDA (`torch.cuda`, `.cuda()`), and it checks in
  the compiler venv that Triton's active backend and the driver's compile target are both `furiosa`.
  There, `kernel[grid](...)` itself runs on the NPU (`compiler/furiosa_compiler/npu.py`).
- L4 runs every graph under `strace` and passes only if the run opened `/dev/rngd/npu3*` and no other
  NPU or GPU device; L4 waits for npu3 when it is busy (up to 30 min) instead of skipping, so a case
  is judged on the NPU or fails.
- L1 (raised IR on MLIR's CPU JIT) and L3 (furiosa-opt's CPU emulation of the same crate) are
  checks of intermediate stages, never results; the GPU runs Triton only to record golden data,
  in the harness venv, whose Triton has no Furiosa backend.

## Generality rules

The compiler must handle Triton code it has never seen, so it may not recognise our corpus:

- Compiler sources (`compiler/` C++ and Python, outside `compiler/tests/`) must not name a corpus kernel
  (any `@triton.jit` function in a case file) or model/case (`resnet101`, `tutorial_softmax`, ...). The
  harness checks this before every run and fails the run on a hit.
- A kernel outside the supported subset must fail compilation with a diagnostic naming the op; silently
  wrong output is a harness failure at L1.
- Progress is the pass rate per corpus layer: micro cases, Triton tutorial kernels (code we did not write),
  held-out kernels, and the models.
- Held-out cases (`harness/cases/heldout_*.py`) are kernels written apart from the corpus, in other
  styles (NCHW layout, 3-D grids, `acc += tl.dot`, transposed stores, online reductions), to measure
  whether the compiler generalizes. A held-out failure is fixed only by a general mechanism that the
  corpus also exercises; a change that only a held-out case motivates does not count.

## Operator matrix (harness/ops)

The model corpus says whether whole networks run; the operator matrix says which Triton operators,
dtypes and shapes do, case by case. It follows SOTA-PNU/lowlevel-api-test (default shape [1, 64, 64],
floats `allclose(rtol=1e-2, atol=1e-2)`, integers exact): 903 cases over

- operators: unary math (abs ... sqrt_rn), binary arithmetic, reductions (sum, max, min, argmax, argmin,
  xor_sum), scans and orderings (cumsum, cumprod, flip, sort, softmax), casts and constructors, shape ops
  (trans, reshape, expand_dims, broadcast_to, permute), tl.dot, tensor descriptors, data-masked stores, program ids, atomics (add/max/min/and/or/xor);
- and the rest of lowlevel-api-test's list (group `extra`): reduce_or, topk, map_elementwise, squeeze/ravel,
  broadcast, rand/randn/randint (and 4x; checked against an exact NumPy Philox), associative_scan, join, split,
  interleave, cat, swizzle2d, umulhi, gather, histogram, NaN-propagating maximum, atomics whose old value is used
  (add, xchg, cas), device_print/assert, tl.range loops (inline_asm_elementwise is GPU assembly: not applicable);
- dtypes: float32, float16, bfloat16, int32, int8 (float8_e4m3fn for dot); fp16/bf16 math functions
  compute in fp32, as Triton requires;
- shapes [B, R, C]: [1, 64, 64], [1, 1, 1024], [4, 32, 128], [2, 7, 33] (masked), [1, 256, 256]; dot
  [M, N, K] = [64, 64, 64], [16, 64, 32], [128, 256, 64]; atomics [1, 8, 64] from 8 programs.

```
compiler/.venv/bin/python -m harness.ops.run --stage raise --jobs 16   # raised IR on the CPU JIT (triage, ~3 min)
compiler/.venv/bin/python -m harness.ops.run --stage npu --jobs 8      # kernel[grid] on npu3: the verdict
compiler/.venv/bin/python -m harness.ops.case reduce.argmax.int8.2x7x33 --stage npu   # one case
```

The npu stage of all cases takes hours. To keep one compiler under it while others relink, run it against a copy of
the Triton package: `cp -a third_party/triton/python/triton <snap>/triton` (replace `backends/furiosa`, a symlink,
with a copy of `compiler/backend`), then `PYTHONPATH=<snap>:compiler compiler/.venv/bin/python -m harness.ops.run ...`
(the backend finds `furiosa_compiler` next to its own file, which the copy moves). `harness.run` takes the same
copy through `HARNESS_TRITON=<snap>`.

Each run writes `harness/out/ops/<stage>.{json,md}`: an operator x dtype table and the failures grouped
by the compiler's diagnostic. At the raise stage (raised IR on the CPU JIT) 905 of the 907 cases pass at 1805f3e;
the other two, int32 sort at 256 and 1024 lanes, exceeded the 900 s limit in codegen and take seconds since f6ad81a.
The npu stage at 1.2.0 (ce414f3): 879 of 907 pass. 2 differ on NaN inputs (`extra.maximum_nan`: the engine's max
propagates NaN where the GPU's `maxnumf` skips it, outside the row-wise class). 26 do not compile: sort (20) and
topk (4) exceeded the step splitter's time (on main since b8ee106 they fail to map in seconds), and `gather_data` (2) maps since 832a9e3.

## Running

```
uv run python -m harness.golden vector_add            # once per case (GPU)
uv run python -m harness.run --impl reference         # all cases, all levels
uv run python -m harness.run --impl reference --cases rmsnorm --levels L2,L3
```

With `--impl compiler` the run first executes the compiler's per-pass lit tests (`unit:` line,
compiler/README.md); a failure there fails the run, like a generality hit.

Cases run `--jobs` at a time (default: one per 16 cores, `FURIOSA_BUILD_SLOTS`). Cargo builds use that
many target dirs under `harness/out/target/{cpu,npu}/slot<i>` (each builds the dependencies once; a crate
returns to its own slot), and runs on npu3 take one lock (`~/.cache/furiosa-npu/npu.lock`, shared with
`kernel[grid]` launches): builds run side by side, the device runs one at a time.

The report is a case × level table (PASS / FAIL / SKIP with the reason and the worst error), also
written to `harness/out/report.json`. L4 waits (up to 30 min) until every PE of npu3 is idle; the harness never
opens another NPU (`FURIOSA_VISIBLE_CHIPS=3`).
