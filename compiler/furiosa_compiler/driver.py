"""Compiles a model file's launch graphs with the Furiosa Triton backend, one kernel per distinct
launch specialization (see compiler/README.md for the output contract).

Each launch is specialized harder than Triton's JIT would: every scalar argument becomes a constexpr
and the grid is recorded, so the backend sees constant shapes, loop bounds and divisors. Pointer
arguments carry their buffer's kind, dtype and shape for the raising pass; buffer names are bound per
launch in graph.json, so one compiled kernel serves every launch of the same shape.
"""

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

TARGET = GPUTarget("furiosa", "rngd", 1)


def load_model(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def is_buffer(value) -> bool:
    return hasattr(value, "kind") and hasattr(value, "shape")


@dataclass
class Kernel:
    symbol: str  # <kernel>_<n>, unique per specialization
    kernel: str
    pointer_args: list
    compiled: object = None  # triton CompiledKernel; .asm holds every stage
    error: str = ""

    def stage(self, name) -> str:
        text = self.compiled.asm[name]
        return text.decode() if isinstance(text, bytes) else str(text)

    def written(self) -> list:
        match = re.search(r"furiosa\.written = \[([^\]]*)\]", self.stage("linalg"))
        return [int(x.split(":")[0]) for x in match.group(1).split(",") if x.strip()] if match else []


def specialize(launch):
    """The ASTSource, backend options, and buffer name per pointer argument of one launch."""
    fn = launch.kernel
    signature, constexprs, attrs, buffers, names, pointer_args = {}, {}, {}, [], [], []
    for i, (name, param) in enumerate(zip(fn.arg_names, fn.params)):
        if name in launch.args:
            value = launch.args[name]
        elif param.has_default:
            value = param.default
        else:
            raise TypeError(f"{fn.__name__}: no value for {name}")
        if is_buffer(value):
            signature[name] = f"*{value.dtype}"
            attrs[(i,)] = [["tt.divisibility", 16]]
            buffers.append({"arg": i, "kind": value.kind, "dtype": value.dtype, "shape": list(value.shape)})
            names.append(value.name)
            pointer_args.append(name)
        else:
            signature[name] = "constexpr"
            constexprs[(i,)] = value
    grid = tuple(launch.grid) + (1,) * (3 - len(launch.grid))
    options = {"grid": grid, "buffers": json.dumps(buffers)}
    return ASTSource(fn, signature, constexprs, attrs), options, names, pointer_args


def steps(structured: str) -> str:
    """The structured kernel split into sub-kernels where no op class maps it whole (compiler/README.md)."""
    from triton._C.libtriton import furiosa

    if structured.startswith("// furiosa codegen failed:"):
        return structured
    try:
        return furiosa.codegen.split_steps(structured)
    except RuntimeError as e:
        return "// furiosa codegen failed: steps\n" + "".join(f"// {line}\n" for line in str(e).splitlines())


def tiled(steps_text: str) -> str:
    """The steps stage with each kernel tiled by its plan into scf.for loops (furiosa-tile), what codegen maps."""
    from triton._C.libtriton import furiosa

    if steps_text.startswith("// furiosa codegen failed:"):
        return steps_text
    try:
        return furiosa.codegen.tile(steps_text)
    except RuntimeError as e:
        return "// furiosa codegen failed: tile\n" + "".join(f"// {line}\n" for line in str(e).splitlines())


def compile_model(model_path: Path, out: Path, keep_going: bool = False) -> dict:
    """Compiles every launch specialization of the model's graphs into ``out`` and writes graph.json."""
    return compile_graphs(load_model(model_path).graphs(), str(model_path), out, keep_going)


def compile_graphs(model_graphs, model: str, out: Path, keep_going: bool = False) -> dict:
    """compile_model for graphs already in hand (a single launch from the NPU launcher, say)."""
    kernels, by_key, counts, graphs = {}, {}, {}, []
    for graph in model_graphs:
        launches = []
        for launch in graph.launches:
            src, options, names, pointer_args = specialize(launch)
            key = (launch.kernel, repr(sorted(src.constants.items(), key=str)), repr(src.signature), options["grid"], options["buffers"])
            if key not in by_key:
                name = launch.kernel.__name__
                symbol = f"{name}_{counts.get(name, 0)}"
                counts[name] = counts.get(name, 0) + 1
                kernel = Kernel(symbol, name, pointer_args)
                try:
                    kernel.compiled = triton.compile(src, target=TARGET, options=options)
                except Exception as e:  # noqa: BLE001 - reported per kernel
                    if not keep_going:
                        raise
                    kernel.error = f"{type(e).__name__}: {e}".splitlines()[0]
                by_key[key] = kernels[symbol] = kernel
            launches.append({"kernel": by_key[key].symbol, "args": names, "grid": list(options["grid"])})
        buffers = {b.name: {"kind": b.kind, "dtype": b.dtype, "shape": list(b.shape)} for b in graph.buffers.values()}
        graphs.append({"name": graph.name, "buffers": buffers, "launches": launches})

    summary = {}
    for symbol, kernel in kernels.items():
        entry = {"kernel": kernel.kernel, "pointer_args": kernel.pointer_args}
        if kernel.error:
            entry["error"] = kernel.error
        else:
            directory = out / "kernels" / symbol
            directory.mkdir(parents=True, exist_ok=True)
            for stage in ("ttir", "linalg", "structured"):
                (directory / f"{stage}.mlir").write_text(kernel.stage(stage))
            steps_text = steps(kernel.stage("structured"))
            (directory / "steps.mlir").write_text(steps_text)
            (directory / "tiled.mlir").write_text(tiled(steps_text))
            entry["written"] = kernel.written()
            entry["linalg"] = f"kernels/{symbol}/linalg.mlir"
        summary[symbol] = entry
    result = {"model": model, "kernels": summary, "graphs": graphs}
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
