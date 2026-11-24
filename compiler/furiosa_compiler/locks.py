"""Locks that the harness, the NPU launcher and parallel agents share: cargo builds run several at a time,
device runs on npu3 one at a time. No dependencies, so both the compiler venv and the harness import it."""

import fcntl
import hashlib
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

CACHE = Path(os.environ.get("FURIOSA_NPU_CACHE", Path.home() / ".cache" / "furiosa-npu"))
# Device locks stay host-wide whatever FURIOSA_NPU_CACHE says: a private build cache must not mean a private lock.
LOCKS = Path.home() / ".cache" / "furiosa-npu"
# Concurrent cargo builds. Each runs the furiosa-opt driver, which spreads one kernel's work over all
# cores but is serial between its phases; 16 cores per build keep the machine busy.
SLOTS = int(os.environ.get("FURIOSA_BUILD_SLOTS", max(1, (os.cpu_count() or 16) // 16)))


@contextmanager
def target_slot(root: Path, crate: Path, slots: int = SLOTS):
    """One of `slots` cargo target dirs under `root`, held until exit: up to `slots` builds run at once,
    and each slot builds the dependencies once. A crate tries its own slot first, where cargo still has
    its last build."""
    root.mkdir(parents=True, exist_ok=True)
    own = int(hashlib.sha256(str(crate).encode()).hexdigest(), 16) % slots
    while True:
        for i in [own] + [i for i in range(slots) if i != own]:
            lock = open(root / f"slot{i}.lock", "w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            with lock:
                yield root / f"slot{i}"
            return
        time.sleep(1)


# SPMD over chips (docs/usage.md): every kernel runs on FURIOSA_CHIPS chips, its parallel work split over them.
CHIPS = int(os.environ.get("FURIOSA_CHIPS", "1"))


def visible_chips() -> list:
    """The NPUs a run opens: FURIOSA_VISIBLE_CHIPS, which names FURIOSA_CHIPS of them (default: the last ones of
    npu0..npu3, so one chip is npu3). Other users' services hold npu0..npu2: a run opens them only when idle
    (wait_idle)."""
    spec = os.environ.get("FURIOSA_VISIBLE_CHIPS") or ",".join(map(str, range(4 - CHIPS, 4)))
    chips = sorted({int(c) for c in spec.split(",") if c.strip()})
    if len(chips) != CHIPS:
        raise ValueError(f"FURIOSA_VISIBLE_CHIPS={spec} names {len(chips)} NPU(s), FURIOSA_CHIPS={CHIPS}")
    return chips


def chips_idle(chips) -> bool:
    """Whether every PE of every NPU in `chips` is idle (an NPU without a status file has nothing to wait for)."""
    for c in chips:
        path = Path(f"/sys/class/rngd_mgmt/rngd!npu{c}mgmt/pe_occupancy")
        if path.exists() and set(path.read_text().split()) - {"0"}:
            return False
    return True


def wait_idle(chips, minutes: int = 30) -> bool:
    """Waits until every NPU in `chips` is idle; never preempts another user's run. Call it holding npu_lock(chips),
    right before the run opens them."""
    for _ in range(minutes * 6):
        if chips_idle(chips):
            return True
        time.sleep(10)
    return chips_idle(chips)


@contextmanager
def npu_lock(chips=(3,)):
    """The NPUs in `chips` to this process alone until exit, locked in order (npu3's lock is npu.lock)."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for c in sorted(chips):
            lock = stack.enter_context(open(LOCKS / ("npu.lock" if c == 3 else f"npu{c}.lock"), "w"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield
