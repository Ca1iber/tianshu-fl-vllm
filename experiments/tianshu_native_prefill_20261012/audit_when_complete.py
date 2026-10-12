# SPDX-License-Identifier: Apache-2.0
"""Read-only waiting, then audit and archive completed official evidence."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('/workspace/benchmark-results/20261012_native_prefill_tianshu')
    stop=datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    result=root/'official4k/result.json'
    while not result.exists():
        if datetime.now(timezone.utc)>=stop:return 1
        time.sleep(5)
    if json.loads(result.read_text()).get('phase')!='complete':return 1
    directory=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(directory/'audit_official.py'),'--output',str(root/'official4k'),'--math-results',str(root/'math')],check=True)
    subprocess.run([sys.executable,str(directory.parent/'archive_completed_pair.py'),'--source',str(root/'official4k'),
                    '--label','native_prefill_official4k_audited_20261012','--single-probe'],check=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
