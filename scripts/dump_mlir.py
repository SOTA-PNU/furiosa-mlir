"""Compiles a model file's launch graphs to one Triton-dialect (TTIR) MLIR module, without torch or a GPU.

    uv run python scripts/dump_mlir.py models/resnet101.py            # writes models/resnet101.triton.mlir

The model file defines ``graphs()``, returning the ``Graph``s to dump. Every distinct kernel
specialization the graphs launch becomes one ``tt.func`` named ``<kernel>_<n>``, specialized exactly
as Triton's JIT specializes it at launch (constexprs, ints equal to 1, 16-divisibility). Comments
before the module list each graph's buffers and launches, so the file carries the whole model.
"""

import importlib.abc
import os
import sys


class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"No module named {name!r} (dump_mlir runs without torch)")


sys.meta_path.insert(0, _NoTorch())
# Keep Triton 3.9's FP fusion default explicit; 3.10 turns it off unless set.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import triton  # noqa: E402
from triton._C.libtriton import ir  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource, make_backend  # noqa: E402
from triton.runtime.jit import create_function_from_signature  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# TTIR comes from Triton's target-independent frontend; the target only selects that pipeline.
TARGET = GPUTarget("cuda", 120, 32)


def load_model(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def specialize(launch, backend):
    """The (options, signature, constexprs, attrs) Triton's JIT would compile ``launch`` with."""
    fn = launch.kernel
    # As in Triton, a kernel parameter shadows a launch option of the same name (e.g. num_stages).
    launch_options = {k: v for k, v in launch.options.items() if k not in launch.args}
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, options = binder(**launch.args, **launch_options)
    return fn._pack_args(backend, launch_options, bound, specialization, options)


def split_ttir(text: str, symbol: str, kernel_name: str):
    """Renames the kernel's function and location aliases to ``symbol``; returns its alias lines
    before the module, the function body lines, and its alias lines after the module."""
    text = text.replace(f"{ROOT}/", "")
    text = re.sub(rf"tt\.func public @{re.escape(kernel_name)}\(", f"tt.func public @{symbol}(", text)
    text = re.sub(r"#loc(\d*)\b", lambda m: f"#{symbol}_loc{m.group(1)}", text)
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("module"))
    end = max(i for i, line in enumerate(lines) if line.startswith("}"))
    if lines[start].rstrip() != "module {":
        raise ValueError(f"unexpected module header in {symbol}: {lines[start]}")
    before = [line for line in lines[:start] if line.strip()]
    after = [line for line in lines[end + 1:] if line.strip()]
    return before, lines[start + 1:end], after


def describe(value):
    if hasattr(value, "kind") and hasattr(value, "shape"):
        return value.name
    return repr(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="model file defining graphs()")
    parser.add_argument("-o", "--out", type=Path, help="default: <model>.triton.mlir next to the model file")
    args = parser.parse_args()
    model_path = args.model.resolve()
    out = args.out or model_path.with_name(model_path.stem + ".triton.mlir")

    model = load_model(model_path)
    graphs = model.graphs()
    backend = make_backend(TARGET)

    symbols = {}  # specialization key -> symbol
    counts = {}
    funcs, aliases_before, aliases_after = [], [], []
    launch_symbols = []
    for graph in graphs:
        for launch in graph.launches:
            options, signature, constexprs, attrs = specialize(launch, backend)
            fn = launch.kernel
            key = (fn, tuple(signature.items()), tuple(sorted(constexprs.items(), key=str)), repr(sorted(attrs.items())))
            if key not in symbols:
                n = counts.get(fn.__name__, 0)
                counts[fn.__name__] = n + 1
                symbol = f"{fn.__name__}_{n}"
                symbols[key] = symbol
                compiled = triton.compile(ASTSource(fn, signature, constexprs, attrs), target=TARGET, options=options.__dict__)
                before, body, after = split_ttir(compiled.asm["ttir"], symbol, fn.__name__)
                consts = ", ".join(f"{fn.arg_names[path[0]]}={value}" for path, value in constexprs.items())
                funcs.append([f"  // {symbol}: {fn.__name__}[{consts}]", *body])
                aliases_before += before
                aliases_after += after
            launch_symbols.append(symbols[key])

    header = [
        f"// {out.name}: generated by scripts/dump_mlir.py from {model_path.relative_to(ROOT)}; do not edit.",
        f"// Triton {triton.__version__} TTIR (Triton dialect, before GPU lowering): "
        f"{len(funcs)} kernel specializations, {sum(len(g.launches) for g in graphs)} launches.",
    ]
    it = iter(launch_symbols)
    for graph in graphs:
        header += ["//", f"// graph {graph.name}", "//   buffers:"]
        for buf in graph.buffers.values():
            header.append(f"//     {buf.kind:<10} {buf.dtype:<4} {'x'.join(map(str, buf.shape)) or 'scalar':<20} {buf.name}")
        header.append(f"//   launches ({len(graph.launches)}):")
        for i, launch in enumerate(graph.launches):
            runtime = [
                f"{name}={describe(value)}"
                for name, value, param in zip(launch.kernel.arg_names, (launch.args.get(n) for n in launch.kernel.arg_names), launch.kernel.params)
                if not param.is_constexpr
            ]
            header.append(f"//     {i:4d}  @{next(it)}  grid={launch.grid}  {' '.join(runtime)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(header + aliases_before + ["module {"] + [line for f in funcs for line in f] + ["}"] + aliases_after) + "\n"
    out.write_text(text)

    # The merged file must parse back as one module holding every function.
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(out), context)
    parsed = sum(1 for line in text.splitlines() if line.lstrip().startswith("tt.func public @"))
    assert parsed == len(funcs), (parsed, len(funcs))
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: {len(funcs)} kernels, "
          f"{len(launch_symbols)} launches, {len(text.splitlines())} lines; parsed {module is not None}")


if __name__ == "__main__":
    main()
