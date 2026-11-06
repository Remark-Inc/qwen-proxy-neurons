"""Bound one explicitly supplied Modal app's total wall time, including its queue."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main(app_id, seconds):
    if not app_id.startswith('ap-') or seconds <= 0:raise ValueError('Explicit app ID and positive cap required')
    cli=str(Path(sys.executable).with_name('modal'))
    while True:
        r=subprocess.run([cli,'app','info',app_id,'--json'],capture_output=True,text=True,timeout=30)
        if r.returncode:
            print('Lifecycle read failed; retrying',flush=True);time.sleep(15);continue
        j=json.loads(r.stdout)
        if j['lifecycle']['state']=='stopped':
            print('App stopped; budget guard complete',flush=True);return
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(j['lifecycle']['created_at'])).total_seconds()
        if elapsed>=seconds:
            print('Stopping explicitly targeted app at wall-time budget cap',flush=True)
            subprocess.run([cli,'app','stop',app_id,'--yes'],check=True,timeout=30);return
        time.sleep(min(30,max(1,seconds-elapsed)))


if __name__=='__main__':main(sys.argv[1],float(sys.argv[2]))
