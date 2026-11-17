"""Runs the operator matrix (harness/ops/cases.py), each case in its own process, and writes
harness/out/ops/<stage>.json and <stage>.md: which dtypes, shapes and operators work.

    compiler/.venv/bin/python -m harness.ops.run --stage raise --jobs 16   # raised IR on the CPU JIT (triage)
    compiler/.venv/bin/python -m harness.ops.run --stage npu --jobs 8      # the verdict on npu3: builds in parallel,
                                                                           # device runs one at a time (npu.py)
    compiler/.venv/bin/python -m harness.ops.run --stage npu --only exp,add --dtypes float32 --shapes 1x64x64
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness.ops.cases import FLOATS, all_cases

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "harness" / "out" / "ops"
COLUMNS = FLOATS + ["int32", "int8", "float8_e4m3fn"]


def run_case(key: str, stage: str, timeout: int) -> dict:
    # Shared build lock: never load a libtriton that compiler/build.sh is relinking.
    cmd = [str(ROOT / "compiler" / "with-build-lock.sh"), sys.executable, "-m", "harness.ops.case", key, "--stage", stage]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"key": key, "status": "ERROR", "detail": f"timed out after {timeout}s"}
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith('{"key"')), None)
    if line:
        result = json.loads(line)
        # The compiler's own diagnostic beats "PassManager::run failed".
        cause = next((l.split("error: ", 1)[1] for l in proc.stderr.splitlines() if "error: furiosa" in l), None)
        if result["status"] == "ERROR" and cause:
            result["detail"] = f"{cause} | {result['detail']}"
        return result
    tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return {"key": key, "status": "ERROR", "detail": f"exited {proc.returncode}: {tail[:300]}"}


def markdown(results: list[dict], stage: str) -> str:
    counts = defaultdict(int)
    cells = defaultdict(lambda: [0, 0])  # (group.op, dtype) -> [passed, total]
    causes = defaultdict(list)
    for r in results:
        group, op, dtype, shape = r["key"].split(".")
        counts[r["status"]] += 1
        cell = cells[(f"{group}.{op}", dtype)]
        cell[1] += 1
        cell[0] += r["status"] == "PASS"
        if r["status"] != "PASS":
            causes[(r["status"], r["detail"].split(" | ")[0][:80] if r["status"] == "ERROR" else "accuracy")].append(r)
    lines = [f"# Operator matrix ({stage})", "",
             f"PASS {counts['PASS']}, FAIL {counts['FAIL']}, ERROR {counts['ERROR']} of {len(results)} cases.", "",
             "Each cell: shapes passing / shapes tested (✓ when all pass).", "",
             "| operator | " + " | ".join(COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
    for op in sorted({k[0] for k in cells}):
        row = []
        for dtype in COLUMNS:
            if (op, dtype) not in cells:
                row.append("")
                continue
            ok, total = cells[(op, dtype)]
            row.append("✓" if ok == total else f"{ok}/{total}")
        lines.append(f"| {op} | " + " | ".join(row) + " |")
    lines += ["", "## Failures by cause", ""]
    for (status, cause), rs in sorted(causes.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- **{status}** {cause} — {len(rs)} cases, e.g. `{rs[0]['key']}`: {rs[0]['detail'][:200]}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["raise", "npu"], default="npu")
    parser.add_argument("--only", default="", help="operators, e.g. exp,add,dot")
    parser.add_argument("--dtypes", default="")
    parser.add_argument("--shapes", default="", help="e.g. 1x64x64,2x7x33")
    parser.add_argument("--jobs", type=int, default=8, help="cases at once (npu3 runs stay one at a time)")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    pick = lambda value, allowed: not allowed or value in allowed.split(",")  # noqa: E731
    cases = [c for c in all_cases() if pick(c.op, args.only) and pick(c.dtype, args.dtypes)
             and pick("x".join(map(str, c.shape)), args.shapes)]
    results = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for i, r in enumerate(pool.map(lambda c: run_case(c.key, args.stage, args.timeout), cases), 1):
            results.append(r)
            print(f"[{i}/{len(cases)}] {r['status']:5} {r['key']}  {r['detail'][:120]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.stage}.json").write_text(json.dumps(results, indent=1) + "\n")
    (OUT / f"{args.stage}.md").write_text(markdown(results, args.stage))
    print(markdown(results, args.stage))


if __name__ == "__main__":
    main()
