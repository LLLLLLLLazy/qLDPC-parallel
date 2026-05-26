from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLIDING_ROOT = ROOT / "SlidingWindowDecoder"
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

if str(SLIDING_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIDING_ROOT))

