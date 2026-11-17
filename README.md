# furiosa-mlir

An MLIR compiler from Triton to the Furiosa RNGD NPU. Models and kernels written in pure Triton run on
the NPU through Triton's own `kernel[grid](...)`: the furiosa backend raises each kernel's Triton dialect
to linalg, structures and tiles it, maps it to Furiosa's tensor ops (fopt), and emits a furiosa-opt Rust
crate that runs on `npu3`. Nothing falls back to a CPU or GPU path (harness/README.md, NPU-only rule).

```
compiler/build.sh                                      # libtriton with the furiosa plugin, and furiosa-mlir-opt
compiler/.venv/bin/python my_kernels.py                # kernel[grid](...) compiles for and runs on the NPU
compiler/furiosa trace models/lenet5.py -o out/lenet5 --npu        # a model launch by launch: CPU reference vs npu3
.venv/bin/python -m harness.run --impl compiler --jobs 8          # the corpus, levels L0-L4
compiler/.venv/bin/python -m harness.ops.run --stage npu --jobs 8 # the operator matrix on the NPU
```

- `compiler/`: the passes (raise to linalg, structure, steps, tile, op classes, fopt, DMA), the Triton backend
  and the crate generator; per-pass lit tests in `compiler/tests/lit` (compiler/README.md).
- `models/`: LeNet-5, AlexNet, VGG16, ResNet-101, MobileNetV3-Large and Qwen3-0.6B in pure Triton.
- `docs/usage.md`: compiling, running and tracing models (`furiosa compile | run | trace`), debugging the stages,
  the binary package (`scripts/package.sh`) and the Docker playground (`docker/`).
- `playground/`: kernels and a notebook to start from.
- `CHANGELOG.md`: the versions (releases on `release/X.Y.Z` branches, tagged `vX.Y.Z`; main is the next `.dev0`).
- `harness/`: the corpus (micro kernels, Triton's own tutorials verbatim, held-out kernels, the models), golden
  data from the GPU, levels L0 (TTIR) to L4 (npu3), and the operator matrix modeled on SOTA-PNU/lowlevel-api-test
  (harness/README.md).
