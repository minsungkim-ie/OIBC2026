"""제출 형식 검증. 하나라도 실패하면 exit 1.  사용: 04_check_submission.py [submissions/v1]"""
import sys
from pathlib import Path

from _bootstrap import log
from oibc.checks import check_submission
from oibc.config import SUB

sub_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SUB
fails, warns, info = check_submission(sub_dir)
for m in info:
    print("  정보:", m)
for m in warns:
    print("  경고:", m)
for m in fails:
    print("  실패:", m)
if fails:
    print(f"검증 실패 ({len(fails)}건)")
    sys.exit(1)
print("검증 통과" + (f" (경고 {len(warns)}건)" if warns else ""))
