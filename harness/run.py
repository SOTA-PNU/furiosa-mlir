"""Judges every case at every level and prints the case x level table (see harness/README.md).

    uv run python -m harness.run --impl reference
    uv run python -m harness.run --impl reference --cases rmsnorm --levels L2,L3
"""

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from harness.golden import GOLDEN
from harness.registry import CASES
from tests.harness import ROOT

sys.path.insert(0, str(ROOT / "compiler"))
from furiosa_compiler.locks import SLOTS, npu_lock, target_slot  # noqa: E402 - dependency-free

LEVELS = ["L0", "L1", "L2a", "L2", "L3", "L4"]
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


def cargo_env(target: str | None, npu: bool) -> dict:
    env = dict(os.environ, **({"CARGO_TARGET_DIR": target} if target else {}))
    env["PATH"] = f"{TOOLS}:{env['PATH']}"
    if not Path("/usr/lib/x86_64-linux-gnu/libclang.so").exists() and Path("/usr/lib/llvm-18/lib").exists():
        env.setdefault("LIBCLANG_PATH", "/usr/lib/llvm-18/lib")
    if npu:
        env["FURIOSA_VISIBLE_CHIPS"] = str(NPU)
    return env


def sh(cmd, cwd, env=None, timeout=3600):
    # Its own session, killed whole on timeout: a grandchild left running would keep what it holds (the
    # build lock compiler/with-build-lock.sh takes, say) and stall every other run.
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return 124, f"timed out after {timeout}s"
    return proc.returncode, out + err


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
        # NaN exactly where golden has it (asin outside [-1, 1], say); the other elements within tolerance.
        o, r = out.float(), ref.float()
        if not torch.equal(o.isnan(), r.isnan()):
            return Result("FAIL", f"{name}: NaN where golden has a number, or the reverse")
        o, r = o[~r.isnan()], r[~r.isnan()]
        err = (o - r).abs().max().item() if r.numel() else 0.0
        scale = r.abs().max().item() if r.numel() else 0.0
        if not err <= tol.atol + tol.rtol * scale:  # also catches inf - inf
            return Result("FAIL", f"{name}: max err {err:.3g} > {tol.atol} + {tol.rtol}*{scale:.3g}")
        if scale and err / scale >= worst:
            worst, worst_name = err / scale, name
    return Result("PASS", f"rel {worst:.2g} @ {worst_name}" if worst_name else "")


def crate_dir(name: str, impl: str) -> Path:
    return ROOT / "harness" / "reference" / name if impl == "reference" else OUT / name / "crate"


def compiler_env() -> dict:
    return dict(os.environ, PYTHONPATH=str(ROOT / "compiler"))


def compiler_cmd(*args, exe: Path = COMPILER_PY) -> list:
    """A compiler-venv command that holds compiler/build.sh's lock shared, so it never loads a
    libtriton that a rebuild is still linking."""
    return [str(ROOT / "compiler" / "with-build-lock.sh"), str(exe), *args]


_compiled = {}


def compile_case(name: str) -> Result:
    """Runs the compiler once per case: kernels, graph.json and the crate under harness/out/<case>/."""
    if name not in _compiled:
        if not COMPILER_PY.exists():
            _compiled[name] = Result("SKIP", "compiler/.venv missing (see compiler/README.md)")
        else:
            cmd = compiler_cmd("-m", "furiosa_compiler", CASES[name].path, "-o", str(OUT / name / "compiler"),
                               "--crate", str(OUT / name / "crate"))
            code, log = sh(cmd, ROOT, compiler_env())
            _compiled[name] = Result("PASS") if code == 0 else Result("FAIL", f"compile: {tail(log)}")
    return _compiled[name]


def build_run_graph(crate: Path, npu: bool) -> tuple[str | None, str]:
    """Builds the crate's run_graph and returns its executable (or None and the build log)."""
    cmd = ["cargo", "furiosa-opt", "build"] if npu else ["cargo", "build"]
    cmd += ["--release", "--bin", "run_graph", "--message-format=json-render-diagnostics"]
    # A few builds at once, each in a target slot whose dependencies are built (harness/README.md). Binaries
    # run from their hashed deps/ path, never target/release/run_graph: every crate's binary has that name
    # (cargo#6313), so resolve it before another build in the slot relinks it.
    with target_slot(OUT / "target" / ("npu" if npu else "cpu"), crate) as target:
        code, log = sh(cmd, crate, cargo_env(str(target), npu))
        exe = None
        for line in log.splitlines():
            if line.startswith("{") and '"executable"' in line:
                msg = json.loads(line)
                if msg.get("executable") and msg.get("target", {}).get("name") == "run_graph":
                    uplifted = Path(msg["executable"])
                    exe = str(next(p for p in (uplifted.parent / "deps").glob("run_graph-*")
                                   if not p.suffix and p.stat().st_ino == uplifted.stat().st_ino))
    return (exe if code == 0 else None), log


def run_graphs(name: str, crate: Path, npu: bool, tol) -> Result:
    """Runs run_graph for every golden graph index and compares its outputs."""
    meta = json.loads((GOLDEN / name / "meta.json").read_text())
    exe, log = build_run_graph(crate, npu)
    if not exe:
        return Result("FAIL", f"build: {tail(log)}")
    worst = []
    with npu_lock() if npu else nullcontext():  # builds run side by side, npu3 runs one at a time
        if npu:  # L4 is the verdict: wait for npu3 rather than skip it.
            for _ in range(180):
                if npu_idle():
                    break
                time.sleep(10)
            else:
                return Result("FAIL", f"npu{NPU} stayed busy for 30 min")
        for graph in meta["graphs"]:
            i = graph["index"]
            out = OUT / name / ("npu" if npu else "cpu") / str(i) / "outputs.safetensors"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.unlink(missing_ok=True)
            cmd = [exe, str(i), str(GOLDEN / name / str(i) / "inputs.safetensors"), str(out)]
            trace = out.parent / "openat.trace"
            if npu:  # record every device the run opens: the proof it ran on npu3
                cmd = ["strace", "-f", "-qq", "-e", "trace=openat", "-o", str(trace)] + cmd
            code, log = sh(cmd, crate, cargo_env(None, npu))
            if code != 0 or not out.exists():
                return Result("FAIL", f"graph {i}: run_graph exited {code}: {tail(log)}")
            if npu:
                text = trace.read_text()
                opened = set(re.findall(r'"/dev/rngd/(npu\d+)', text))
                if opened != {f"npu{NPU}"} or '"/dev/nvidia' in text:
                    return Result("FAIL", f"graph {i}: did not run on npu{NPU} alone (opened {sorted(opened) or 'no NPU'}"
                                          f"{', a GPU' if '/dev/nvidia' in text else ''})")
            result = compare(out, GOLDEN / name / str(i) / "outputs.safetensors", tol)
            top1 = CASES[name].top1
            if result.status == "PASS" and top1:
                got, want = load_file(out)[top1].float(), load_file(GOLDEN / name / str(i) / "outputs.safetensors")[top1].float()
                if not torch.equal(got.reshape(-1, got.shape[-1]).argmax(-1), want.reshape(-1, want.shape[-1]).argmax(-1)):
                    result = Result("FAIL", f"{top1}: top-1 differs from golden")
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
        # Two checks: every launch on golden inputs (one ulp), then the whole graph chained.
        verdicts = []
        for module, tol in (("furiosa_compiler.l1_launch", case.exact), ("furiosa_compiler.l1", case.graph or case.exact)):
            cmd = compiler_cmd("-m", module, str(OUT / name / "compiler"), str(GOLDEN / name),
                               "--rtol", str(tol.rtol), "--atol", str(tol.atol))
            code, log = sh(cmd, ROOT, compiler_env())
            try:
                verdict = json.loads(next(line for line in reversed(log.splitlines()) if line.startswith('{"status"')))
            except (ValueError, StopIteration):
                return Result("FAIL", f"{module} exited {code}: {tail(log)}")
            if verdict["status"] != "PASS":
                kind = "per launch" if module.endswith("l1_launch") else "graph"
                return Result("FAIL", f"{kind}: {verdict.get('detail', '')}")
            verdicts.append(verdict.get("detail", ""))
        return Result("PASS", "; ".join(d for d in verdicts if d))
    if impl == "compiler":
        compiled = compile_case(name)
        if compiled.status != "PASS":
            return compiled
    crate = crate_dir(name, impl)
    if not (crate / "Cargo.toml").exists():
        return Result("SKIP", f"no {impl} crate")
    failed = crate / "CODEGEN_FAILED"
    if failed.exists():  # the compiler wrote a crate that only carries its codegen diagnostic
        if level not in ("L2a", "L2"):
            return Result("SKIP", "L2 failed: codegen left kernels unmapped")
        lines = failed.read_text().strip().replace(str(ROOT) + "/", "").splitlines()
        # One line per failing kernel (or step of one): "<symbol>[ step k (...)]: reason".
        unmapped = {line.split(":")[0].split(" ")[0] for line in lines if line.strip()}
        total = len(json.loads((OUT / name / "compiler" / "graph.json").read_text())["kernels"])
        return Result("FAIL", f"codegen: {len(unmapped)}/{total} kernels unmapped; {lines[0][:300]}")
    if level == "L2a":  # every kernel mapped and the crate written: seconds, no cargo build
        return Result("PASS")
    if level == "L2":
        exe, log = build_run_graph(crate, npu=True)
        code = 0 if exe else 1
        return Result("PASS") if code == 0 else Result("FAIL", tail(log))
    if not (GOLDEN / name / "meta.json").exists():
        return Result("SKIP", f"no golden (run: python -m harness.golden {name})")
    if level == "L3":
        return run_graphs(name, crate, npu=False, tol=case.npu)
    if level == "L4":
        return run_graphs(name, crate, npu=True, tol=case.npu)
    raise ValueError(level)


def generality_violations() -> list:
    """Corpus kernel and case names that appear in compiler sources (see harness/README.md)."""
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


# Product code that would run Triton anywhere but the NPU: another Triton target, another driver, CUDA.
FALLBACK = re.compile(r'GPUTarget\(\s*"(cuda|hip)|TRITON_DEFAULT_BACKEND"\]\s*=\s*"(?!furiosa)|driver\.set_active|'
                      r'torch\.cuda|\.cuda\(|device\s*=\s*"cuda"')


def npu_only_violations() -> list:
    """Ways the compiler could lower Triton to a CPU or GPU instead of the NPU (harness/README.md)."""
    hits = []
    for path in sorted((ROOT / "compiler").rglob("*")):
        rel = path.relative_to(ROOT)
        if path.suffix not in {".py", ".cpp", ".cc", ".h"} or rel.parts[1] in ("tests", ".venv") or "__pycache__" in rel.parts:
            continue
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if FALLBACK.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:100]}")
    if COMPILER_PY.exists():
        probe = ("import triton; from furiosa_compiler.driver import TARGET; "
                 "print(triton.runtime.driver.active.get_current_target().backend, TARGET.backend)")
        code, log = sh(compiler_cmd("-c", probe), ROOT, compiler_env())
        if log.strip().splitlines()[-1:] != ["furiosa furiosa"]:
            hits.append(f"compiler venv: Triton's active backend / compile target is not furiosa: {tail(log)}")
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--impl", choices=["reference", "compiler"], default="reference")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--jobs", type=int, default=SLOTS, help="cases run at once (npu3 runs stay one at a time)")
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
    npu_hits = npu_only_violations() if args.impl == "compiler" else []
    print("npu-only: " + ("FAIL\n  " + "\n  ".join(npu_hits) if npu_hits else "PASS"))
    unit_failed = False
    if args.impl == "compiler" and COMPILER_PY.exists():
        # Per-pass lit tests (compiler/tests/lit): seconds, and they name the pass that broke.
        code, log = sh(compiler_cmd("-q", str(ROOT / "compiler" / "tests" / "lit"), exe=COMPILER_PY.parent / "lit"),
                       ROOT, compiler_env())
        unit_failed = code != 0
        print("unit: " + ("FAIL\n" + log.strip() if unit_failed else "PASS"))

    report = {name: {} for name in names}
    width = max(len(n) for n in names)
    print(f"{'case':<{width}}  " + "  ".join(levels))

    def run_case(name):
        (OUT / name).mkdir(parents=True, exist_ok=True)
        with open(OUT / name / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # concurrent runs of a case share its harness/out folder
            for level in levels:
                result = judge(name, level, args.impl)
                report[name][level] = {"status": result.status, "detail": result.detail}
                print(f"{name:<{width}}  {level}: {result.cell()}", flush=True)

    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        list(pool.map(run_case, names))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps({"impl": args.impl, "generality": hits, "results": report}, indent=2) + "\n")

    print()
    print(f"{'case':<{width}}  " + "  ".join(f"{lv:<4}" for lv in levels))
    for name in names:
        print(f"{name:<{width}}  " + "  ".join(f"{report[name][lv]['status']:<4}" for lv in levels))
    # Progress is the pass rate per corpus layer (harness/README.md).
    def layer(name):
        if CASES[name].path.startswith("models/"):
            return "models"
        prefix = name.split("_")[0]
        return prefix if prefix in ("tutorial", "heldout") else "micro"
    print()
    for group in ("micro", "tutorial", "heldout", "models"):
        members = [n for n in names if layer(n) == group]
        if members:
            rates = [f"{lv} {sum(report[n][lv]['status'] == 'PASS' for n in members)}/{len(members)}" for lv in levels]
            print(f"{group:<8}  " + "  ".join(rates))
    failed = bool(hits) or bool(npu_hits) or unit_failed or any(r["status"] == "FAIL" for levels_ in report.values() for r in levels_.values())
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
