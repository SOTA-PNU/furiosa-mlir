# Playground

Write a `@triton.jit` kernel in a `.py` file here and run it with the compiler venv:

```
compiler/.venv/bin/python playground/vector_add.py
```

In this venv a launch (`kernel[grid](...)` on CPU torch tensors) compiles the kernel through every stage,
builds a furiosa-opt crate and runs it on npu3; the tensors the kernel writes come back to the host.
A kernel the compiler cannot map raises with its diagnostic: nothing falls back to the CPU or GPU.

The first launch of each specialization (kernel, constexprs, grid, tensor dtypes/shapes) builds the crate;
later launches reuse it. Its stages land in `~/.cache/furiosa-npu/<kernel>-<hash>/`, in pipeline order:

| File | Stage |
|---|---|
| `out/kernels/<symbol>/ttir.mlir` | Triton IR |
| `linalg.mlir` | raised to whole-tensor linalg |
| `structured.mlir`, `steps.mlir` | structured linalg, split into steps |
| `tiled.mlir`, `ftile.mlir`, `fopt.mlir` | tiled, then the furiosa-opt dialect |
| `crate/` | the generated Rust crate |

To compile without running, call `triton.compile` with the furiosa target and the launch (grid, buffer shapes):
`playground/compile_only.py` prints the `ttir`, `linalg`, `structured`, `fopt` and `rust` stages in seconds.

`playground.ipynb` walks through a kernel, its stages, and a model compiled, run and traced with the `furiosa`
CLI; `docker/run.sh` serves it in JupyterLab with npu3 (docs/usage.md). `compiler/README.md` describes each stage. After a C++ change to the compiler, rebuild with `compiler/build.sh`.
