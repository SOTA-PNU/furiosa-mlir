"""Judges every case at every level and prints the case x level table (see harness/README.md).

    uv run python -m harness.run --impl reference
    uv run python -m harness.run --impl reference --cases rmsnorm --levels L2,L3
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from harness.golden import GOLDEN
from harness.registry import CASES
from tests.harness import ROOT

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
OUT = ROOT / "harness" / "out"
NPU = 3  # the only NPU the harness may open
TOOLS = ROOT / ".tools" / "bin"
COMPILER_PY = ROOT / "compiler" / ".venv" / "bin" / "python"  # source-built Triton with the furiosa plugin


@dataclass
class Result:
    status: str  # PASS, FAIL, SKIP
    detail: str = ""

    def cell(self):
        return self.status if not self.detail else f"{self.status} ({self.detail})"


def cargo_env(target: str, npu: bool) -> dict:
    env = dict(os.environ, CARGO_TARGET_DIR=target)
    env["PATH"] = f"{TOOLS}:{env['PATH']}"
    if not Path("/usr/lib/x86_64-linux-gnu/libclang.so").exists() and Path("/usr/lib/llvm-18/lib").exists():
        env.setdefault("LIBCLANG_PATH", "/usr/lib/llvm-18/lib")
    if npu:
        env["FURIOSA_VISIBLE_CHIPS"] = str(NPU)
    return env


def sh(cmd, cwd, env=None, timeout=3600):
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout + proc.stderr)


def tail(log: str, n: int = 3) -> str:
    lines = [line for line in log.strip().splitlines() if line.strip()]
    return " | ".join(lines[-n:])[-300:]


def npu_idle() -> bool:
    path = Path(f"/sys/class/rngd_mgmt/rngd!npu{NPU}mgmt/pe_occupancy")
    return path.exists() and set(path.read_text().split()) == {"0"}


def compare(out_path: Path, golden_path: Path, tol) -> Result:
    got, want = load_file(out_path), load_file(golden_path)
    missing = sorted(set(want) - set(got))
    if missing:
        return Result("FAIL", f"outputs missing {missing[:3]}")
    worst, worst_name = 0.0, ""
    for name, ref in want.items():
        out = got[name]
        if out.shape != ref.shape or out.dtype != ref.dtype:
            return Result("FAIL", f"{name}: got {out.dtype}{list(out.shape)}, want {ref.dtype}{list(ref.shape)}")
        if not ref.is_floating_point():
            if not torch.equal(out, ref):
                return Result("FAIL", f"{name}: {(out != ref).sum().item()} integer mismatches")
            continue
        err = (out.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        if not err <= tol.atol + tol.rtol * scale:  # also catches NaN
            return Result("FAIL", f"{name}: max err {err:.3g} > {tol.atol} + {tol.rtol}*{scale:.3g}")
        if scale and err / scale >= worst:
            worst, worst_name = err / scale, name
    return Result("PASS", f"rel {worst:.2g} @ {worst_name}" if worst_name else "")


def crate_dir(name: str, impl: str) -> Path:
    return ROOT / "harness" / "reference" / name if impl == "reference" else OUT / name / "crate"


def compiler_env() -> dict:
    return dict(os.environ, PYTHONPATH=str(ROOT / "compiler"))


_compiled = {}


def compile_case(name: str) -> Result:
    """Runs the compiler once per case: kernels, graph.json and the crate under harness/out/<case>/."""
    if name not in _compiled:
        if not COMPILER_PY.exists():
            _compiled[name] = Result("SKIP", "compiler/.venv missing (see compiler/README.md)")
        else:
            cmd = [str(COMPILER_PY), "-m", "furiosa_compiler", CASES[name].path, "-o", str(OUT / name / "compiler"),
                   "--crate", str(OUT / name / "crate")]
            code, log = sh(cmd, ROOT, compiler_env())
            _compiled[name] = Result("PASS") if code == 0 else Result("FAIL", f"compile: {tail(log)}")
    return _compiled[name]


def run_graphs(name: str, crate: Path, npu: bool, tol) -> Result:
    """Runs run_graph for every golden graph index and compares its outputs."""
    meta = json.loads((GOLDEN / name / "meta.json").read_text())
    target = str(crate / "target" / ("npu" if npu else "cpu"))
    worst = []
    for graph in meta["graphs"]:
        i = graph["index"]
        out = OUT / name / ("npu" if npu else "cpu") / str(i) / "outputs.safetensors"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        args = ["--release", "--bin", "run_graph", "--", str(i), str(GOLDEN / name / str(i) / "inputs.safetensors"), str(out)]
        cmd = ["cargo", "furiosa-opt", "run", *args] if npu else ["cargo", "run", *args]
        code, log = sh(cmd, crate, cargo_env(target, npu))
        if code != 0 or not out.exists():
            return Result("FAIL", f"graph {i}: run_graph exited {code}: {tail(log)}")
        result = compare(out, GOLDEN / name / str(i) / "outputs.safetensors", tol)
        if result.status != "PASS":
            result.detail = f"graph {i}: {result.detail}"
            return result
        worst.append(result.detail)
    return Result("PASS", "; ".join(d for d in worst if d))


def judge(name: str, level: str, impl: str) -> Result:
    case = CASES[name]
    if not (ROOT / case.path).exists():
        return Result("SKIP", f"{case.path} missing")
    if level == "L0":
        out = OUT / name / f"{name}.triton.mlir"
        out.parent.mkdir(parents=True, exist_ok=True)
        code, log = sh([sys.executable, "scripts/dump_mlir.py", case.path, "-o", str(out)], ROOT)
        return Result("PASS") if code == 0 else Result("FAIL", tail(log))
    if level == "L1":
        if impl != "compiler":
            return Result("SKIP", "L1 judges the compiler")
        compiled = compile_case(name)
        if compiled.status != "PASS":
            return compiled
        if not (GOLDEN / name / "meta.json").exists():
            return Result("SKIP", f"no golden (run: python -m harness.golden {name})")
        tol = case.exact
        cmd = [str(COMPILER_PY), "-m", "furiosa_compiler.l1", str(OUT / name / "compiler"), str(GOLDEN / name),
               "--rtol", str(tol.rtol), "--atol", str(tol.atol)]
        code, log = sh(cmd, ROOT, compiler_env())
        try:
            verdict = json.loads(log.strip().splitlines()[-1])
            return Result(verdict["status"], verdict.get("detail", ""))
        except (ValueError, IndexError, KeyError):
            return Result("FAIL", f"l1 exited {code}: {tail(log)}")
    if impl == "compiler":
        compiled = compile_case(name)
        if compiled.status != "PASS":
            return compiled
    crate = crate_dir(name, impl)
    if not (crate / "Cargo.toml").exists():
        return Result("SKIP", f"no {impl} crate")
    if level == "L2":
        # A target dir per crate: every crate's binary is named run_graph, so a shared one could run
        # another crate's binary (cargo#6313).
        target = str(crate / "target" / "npu")
        code, log = sh(["cargo", "furiosa-opt", "build", "--release", "--bin", "run_graph"], crate, cargo_env(target, npu=True))
        return Result("PASS") if code == 0 else Result("FAIL", tail(log))
    if not (GOLDEN / name / "meta.json").exists():
        return Result("SKIP", f"no golden (run: python -m harness.golden {name})")
    if level == "L3":
        return run_graphs(name, crate, npu=False, tol=case.npu)
    if level == "L4":
        if not npu_idle():
            return Result("SKIP", f"npu{NPU} busy")
        return run_graphs(name, crate, npu=True, tol=case.npu)
    raise ValueError(level)


def generality_violations() -> list:
    """Corpus kernel and case names that appear in compiler sources (see harness/README.md)."""
    import re

    from tests.harness import load_model

    names = set(CASES) | {"resnet", "mobilenet", "qwen", "lenet", "alexnet", "vgg"}
    for case in CASES.values():
        if (ROOT / case.path).exists():
            # The kernels a graph launches; jit helpers with generic names (tanh) are not corpus-specific.
            names |= {launch.kernel.__name__ for graph in load_model(case.path).graphs() for launch in graph.launches}
    generic = {"vector_add", "rmsnorm", "matmul", "embedding"}  # op-class words, legitimate in a compiler
    names -= generic
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\w*", re.IGNORECASE)
    hits = []
    for path in sorted((ROOT / "compiler").rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or path.suffix not in {".py", ".cpp", ".cc", ".h", ".td", ".txt"}:
            continue
        if rel.parts[1] in ("tests", ".venv") or "__pycache__" in rel.parts:
            continue
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if (m := pattern.search(line)):
                hits.append(f"{rel}:{lineno}: {m.group(0)}")
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--impl", choices=["reference", "compiler"], default="reference")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--levels", default=",".join(LEVELS))
    args = parser.parse_args()
    names = args.cases.split(",")
    levels = args.levels.split(",")
    unknown = [n for n in names if n not in CASES] + [lv for lv in levels if lv not in LEVELS]
    if unknown:
        sys.exit(f"unknown {unknown}")

    hits = generality_violations()
    if hits:
        print("generality: FAIL — compiler sources name corpus kernels or cases:")
        for hit in hits[:20]:
            print("  " + hit)
    else:
        print("generality: PASS")

    report = {}
    width = max(len(n) for n in names)
    print(f"{'case':<{width}}  " + "  ".join(levels))
    for name in names:
        report[name] = {}
        for level in levels:
            result = judge(name, level, args.impl)
            report[name][level] = {"status": result.status, "detail": result.detail}
            print(f"{name:<{width}}  {level}: {result.cell()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps({"impl": args.impl, "generality": hits, "results": report}, indent=2) + "\n")

    print()
    print(f"{'case':<{width}}  " + "  ".join(f"{lv:<4}" for lv in levels))
    for name in names:
        print(f"{name:<{width}}  " + "  ".join(f"{report[name][lv]['status']:<4}" for lv in levels))
    failed = bool(hits) or any(r["status"] == "FAIL" for levels_ in report.values() for r in levels_.values())
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
