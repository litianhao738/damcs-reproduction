"""Episode checkpoints: save the whole running state, resume after an interruption.

A checkpoint is one pickle of the environment, the agents (with their memories),
the intervention controller, the loop counters, model usage and the global random
states. Pickle keeps shared references intact (an agent's ``env`` is the restored
environment), so the resumed run continues from exactly the saved step.

A checkpoint is tied to the code that wrote it: resuming with a different source
hash is refused, because pickled objects depend on the class definitions and the
results would mix two code versions. Checkpoints are only written while no human
review is pending, so a resumed run never inherits a half-finished review.
"""

import json
import pickle
from pathlib import Path
import random
import re

import numpy as np

CHECKPOINT_NAME = "checkpoint.pkl"
CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    """A checkpoint cannot be used to resume this run."""


def checkpoint_path(directory):
    return Path(directory) / CHECKPOINT_NAME


def save_checkpoint(directory, state):
    """Write ``state`` atomically: a crash while saving keeps the previous checkpoint."""
    path = checkpoint_path(directory)
    temporary = path.with_suffix(".tmp")
    payload = {
        "version": CHECKPOINT_VERSION,
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        **state,
    }
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return path


def load_checkpoint(directory, source_sha256=None):
    """Return the saved state, or ``None`` when there is no checkpoint.

    Restores the global random states. Raises :class:`CheckpointError` when the
    checkpoint was written by another checkpoint format or another code version.
    """
    path = checkpoint_path(directory)
    if not path.exists():
        return None
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if state.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError(f"{path} uses checkpoint format {state.get('version')}, "
                              f"this code reads {CHECKPOINT_VERSION}.")
    if source_sha256 is not None and state.get("source_sha256") != source_sha256:
        raise CheckpointError(f"{path} was written by different code (source {state.get('source_sha256')}, "
                              f"now {source_sha256}); resuming would mix two versions. "
                              "Rerun the episode from the start instead.")
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    return state


def remove_checkpoint(directory):
    checkpoint_path(directory).unlink(missing_ok=True)


def _keep_lines_up_to(path, last_step):
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        step = json.loads(line).get("step")
        if step is None or step <= last_step:
            kept.append(line)
    path.write_text("".join(item + "\n" for item in kept), encoding="utf-8")


def trim_after_checkpoint(directory, last_step, requests_issued):
    """Drop what an interrupted run logged after the checkpoint.

    Keeps log lines for steps up to ``last_step``, review files up to the number
    of requests issued at the checkpoint, and failure snapshots up to
    ``last_step``. The resumed run then writes those steps again, once.
    """
    directory = Path(directory)
    for name in ("events.jsonl", "coordination.jsonl"):
        _keep_lines_up_to(directory / name, last_step)
    for path in (directory / "reviews").glob("review_*"):
        match = re.match(r"review_(\d{4})", path.name)
        if match and int(match.group(1)) > requests_issued:
            path.unlink()
    for path in (directory / "failures").glob("step_*.json"):
        if int(path.stem.split("_")[1]) > last_step:
            path.unlink()
