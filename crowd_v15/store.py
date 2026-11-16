"""On-disk state for a v1.5 coordinator.

Weights live in one safetensors file plus a ``status.json`` that a browser can
read without touching the binary.  Both are written atomically so a crash mid
round leaves the previous consistent pair in place -- a coordinator that serves
half-updated weights teaches every client a different model.

The digest function is the interesting one.  ``baseSha256`` in the upload header
binds a gradient to the weights it was measured from, so the client and the
coordinator must compute the *same* digest from the *same* tensors on different
machines.  It is therefore defined over a canonical serialisation: names sorted,
float32 rounded to bfloat16 (the dtype the model is trained in), concatenated.
A digest over the safetensors file itself would differ for any container
change -- a re-save, a compression switch, a different writer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

WEIGHTS = "weights.safetensors"
STATUS = "status.json"
SUBMISSIONS = "submissions.jsonl"


def weights_digest(tensors: dict[str, Any]) -> str:
    from .codec import _f32_to_bf16

    digest = hashlib.sha256()
    for name in sorted(tensors):
        arr = np.ascontiguousarray(np.asarray(tensors[name], dtype=np.float32).reshape(-1))
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_f32_to_bf16(arr).tobytes())
    return digest.hexdigest()


class StoreError(RuntimeError):
    """The on-disk state is missing or inconsistent."""


class ModelStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.root / WEIGHTS
        self.status_path = self.root / STATUS
        self.log_path = self.root / SUBMISSIONS

    # ---- weights ---------------------------------------------------------

    def exists(self) -> bool:
        return self.weights_path.is_file() and self.status_path.is_file()

    def load_weights(self) -> dict[str, np.ndarray]:
        if not self.weights_path.is_file():
            raise StoreError(f"no weights at {self.weights_path}")
        from safetensors.numpy import load_file

        return {k: v.astype(np.float32, copy=False) for k, v in load_file(str(self.weights_path)).items()}

    def save_weights(self, tensors: dict[str, np.ndarray]) -> str:
        """Atomically replace the weights; returns the new digest."""
        from safetensors.numpy import save_file

        clean = {
            k: np.ascontiguousarray(np.asarray(v, dtype=np.float32))
            for k, v in tensors.items()
        }
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".weights-", suffix=".safetensors")
        os.close(fd)
        try:
            save_file(clean, tmp)
            os.replace(tmp, self.weights_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return weights_digest(clean)

    def weights_bytes(self) -> bytes:
        if not self.weights_path.is_file():
            raise StoreError(f"no weights at {self.weights_path}")
        return self.weights_path.read_bytes()

    # ---- status ----------------------------------------------------------

    def load_status(self) -> dict[str, Any]:
        if not self.status_path.is_file():
            return {}
        with open(self.status_path, encoding="utf-8") as fh:
            return json.load(fh)

    def save_status(self, status: dict[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".status-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(status, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, self.status_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def manifest(self, model: dict[str, Any]) -> dict[str, Any]:
        """What a client needs before it can train: shapes, dtype and the base digest."""
        tensors = self.load_weights()
        return {
            "schema": "crowd-v15-model-1",
            "name": model.get("name", "crowd-v1.5-dust"),
            "route": model.get("route", "dust"),
            "dtype": "float32",
            "params": int(sum(int(np.asarray(t).size) for t in tensors.values())),
            "baseSha256": weights_digest(tensors),
            "bytes": int(self.weights_path.stat().st_size),
            "names": sorted(tensors),
            "sizes": [int(np.asarray(tensors[n]).size) for n in sorted(tensors)],
            "shapes": {n: list(np.asarray(tensors[n]).shape) for n in sorted(tensors)},
        }

    # ---- audit trail -----------------------------------------------------

    def append_submission(self, record: dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["ModelStore", "StoreError", "WEIGHTS", "STATUS", "SUBMISSIONS",
           "weights_digest"]
