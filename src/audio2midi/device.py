"""Which accelerator the machine really has: an NVIDIA GPU (CUDA), an Apple GPU (MPS) or just the CPU.

The answer comes from PyTorch itself, run in the engine environment, so it tells what the models will
use and not just what hardware exists (an NVIDIA card with an old driver counts as CPU here).
Set AUDIO2MIDI_DEVICE=cpu to force CPU, e.g. to try what a machine without a GPU would do.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass

_PROBE = r"""
import json
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out = {"kind": "cuda", "name": p.name, "vram_gb": round(p.total_memory / 2**30, 1)}
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        out = {"kind": "mps", "name": "Apple GPU", "vram_gb": None}
    else:
        out = {"kind": "cpu", "name": "", "vram_gb": None}
except Exception as e:
    out = {"kind": "cpu", "name": "", "vram_gb": None, "error": str(e)[:200]}
print(json.dumps(out))
"""


@dataclass(frozen=True)
class Device:
    kind: str = "cpu"             # "cuda", "mps" or "cpu"
    name: str = ""
    vram_gb: float | None = None

    @property
    def gpu(self) -> bool:
        return self.kind != "cpu"

    @property
    def torch_device(self) -> str:
        """The value for `--device` in the engine scripts."""
        return self.kind

    @property
    def label(self) -> str:
        if not self.gpu:
            return "CPU (sin GPU compatible)"
        mem = f", {self.vram_gb:g} GB" if self.vram_gb else ""
        return f"GPU: {self.name}{mem}"


def probe(python_exe: str, timeout: float = 90) -> Device:
    """Ask the engine environment what it can use. Takes a few seconds (it imports torch)."""
    if os.environ.get("AUDIO2MIDI_DEVICE", "").lower() == "cpu":
        return Device()
    try:
        r = subprocess.run([python_exe, "-c", _PROBE], capture_output=True, text=True, timeout=timeout,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        info = json.loads(r.stdout.strip().splitlines()[-1])
        return Device(info["kind"], info.get("name", ""), info.get("vram_gb"))
    except Exception:  # noqa: BLE001 - any failure just means "use the CPU"
        return Device()
