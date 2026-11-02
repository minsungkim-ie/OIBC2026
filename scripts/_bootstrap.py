"""스크립트 공통: src 경로 추가, UTF-8 콘솔, 시드, 로그."""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np  # noqa: E402

from oibc.config import SEED  # noqa: E402

random.seed(SEED)
np.random.seed(SEED)

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)
