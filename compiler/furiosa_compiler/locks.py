"""Locks that the harness, the NPU launcher and parallel agents share: cargo builds run several at a time,
device runs on npu3 one at a time. No dependencies, so both the compiler venv and the harness import it."""

import fcntl
import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

CACHE = Path(os.environ.get("FURIOSA_NPU_CACHE", Path.home() / ".cache" / "furiosa-npu"))
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


@contextmanager
def npu_lock():
    """npu3 to this process alone until exit."""
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / "npu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
