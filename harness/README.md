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

L3 and L4 compare with the case's NPU tolerance, which allows for the dtype policy the NPU forces
(no fp32 or fp16 contraction: dots run in bf16 with fp32 accumulation). A match means
`max|out - golden| <= atol + rtol * max|golden|` for every output.

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

## Generality rules

The compiler must handle Triton code it has never seen, so it may not recognise our corpus:

- Compiler sources (`compiler/` C++ and Python, outside `compiler/tests/`) must not name a corpus kernel
  (any `@triton.jit` function in a case file) or model/case (`resnet101`, `tutorial_softmax`, ...). The
  harness checks this before every run and fails the run on a hit.
- A kernel outside the supported subset must fail compilation with a diagnostic naming the op; silently
  wrong output is a harness failure at L1.
- Progress is the pass rate per corpus layer: micro cases, Triton tutorial kernels (code we did not write),
  and the models.

## Running

```
uv run python -m harness.golden vector_add            # once per case (GPU)
uv run python -m harness.run --impl reference         # all cases, all levels
uv run python -m harness.run --impl reference --cases rmsnorm --levels L2,L3
```

The report is a case × level table (PASS / FAIL / SKIP with the reason and the worst error), also
written to `harness/out/report.json`. L4 is skipped unless every PE of npu3 is idle; the harness never
opens another NPU (`FURIOSA_VISIBLE_CHIPS=3`).
