"""01 → 04 순서 실행. 실패하면 중단."""
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
t0 = time.time()
for s in ["01_validate.py", "02_optimize_composition.py", "03_train_final_predict.py", "04_check_submission.py"]:
    print(f"\n##### {s} #####", flush=True)
    r = subprocess.run([sys.executable, str(here / s)])
    if r.returncode != 0:
        sys.exit(r.returncode)
print(f"\n전체 소요 {time.time() - t0:.0f}s")
