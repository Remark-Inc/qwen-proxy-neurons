"""Launch one http:// path phase with a ledger reservation and an app wall-time guard.

Usage: python -m qwen_lab.http_launch LOG_DIR CAP_SECONDS --phase ... --run-id ...
The guard stops the explicitly launched app once its whole lifetime, including
queueing and startup, reaches CAP_SECONDS. Estimates are not invoices.
"""
import contextlib,datetime,fcntl,json,re,subprocess,sys,time
from pathlib import Path

RATE=.001261+4*.0000131+96*.00000222
LEDGER=Path('data/http/budget.json')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextlib.contextmanager
def locked(path=LEDGER):
    """Serialize ledger read-modify-write across concurrently running launchers."""
    Path('runs').mkdir(exist_ok=True)  # gitignored; keeps the lock file out of the mounted data
    with open('runs/.http-ledger.lock','w') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield json.loads(path.read_text())
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def committed(ledger):
    return sum(j.get('app_wallclock_estimate_usd',j['reservation_usd']) for j in ledger['jobs'] if j['status']!='not_started')


def main(log_dir,cap_seconds,*args,ledger_path=LEDGER,app_file='http_path_app.py'):
    cap=float(cap_seconds);run_id=args[list(args).index('--run-id')+1]
    with locked(ledger_path) as ledger:
        if not ledger.get('authorized'):raise ValueError('Phase spending not authorized in the ledger')
        if any(j['run_id']==run_id for j in ledger['jobs']):raise ValueError('Run ID already used')
        if committed(ledger)+cap*RATE>ledger['phase_cap_usd']:raise ValueError('Reservation would exceed the phase cap')
        job=dict(run_id=run_id,args=list(args),cap_seconds=cap,reservation_usd=cap*RATE,reserved_utc=now(),status='reserved')
        ledger['jobs'].append(job);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    log=Path(log_dir)/f'{run_id}.log';py=sys.executable;app=guard=None
    with log.open('w') as f:
        p=subprocess.Popen([py,'-m','modal','run',app_file,*args],stdout=f,stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(5)
            if app is None and (m:=re.search(r'\bap-[A-Za-z0-9]{10,}',log.read_text())):
                app=m.group()
                guard=subprocess.Popen([py,'-m','qwen_lab.modal_budget_guard',app,str(cap)],
                                       stdout=(Path(log_dir)/f'{run_id}-guard.log').open('w'),stderr=subprocess.STDOUT)
    if guard:guard.wait(timeout=600)
    info={}
    if app:
        for _ in range(20):
            r=subprocess.run([str(Path(py).with_name('modal')),'app','info',app,'--json'],capture_output=True,text=True,timeout=60)
            if r.returncode==0 and json.loads(r.stdout)['lifecycle']['state']=='stopped':info=json.loads(r.stdout);break
            time.sleep(15)
    result=Path('runs')/run_id/'result.json';out=json.loads(result.read_text()) if result.exists() else None
    with locked(ledger_path) as ledger:
        job=next(j for j in ledger['jobs'] if j['run_id']==run_id)
        record(job,p.returncode,app,info,out,result)
        ledger['phase_estimate_usd']=committed(ledger);ledger['research_estimate_usd']=ledger['prior_research_estimate_usd']+ledger['phase_estimate_usd']
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps({k:job.get(k) for k in ['run_id','status','app_id','app_wallclock_seconds','app_wallclock_estimate_usd','measured_compute_usd','complete']}))


def record(job,code,app,info,out,result):
    job.update(exit_code=code,app_id=app,finished_utc=now(),status='completed' if code==0 else 'failed')
    if info:
        life=info['lifecycle'];start=datetime.datetime.fromisoformat(life['created_at']);stop=datetime.datetime.fromisoformat(life['stopped_at'])
        seconds=(stop-start).total_seconds()
        job.update(app_created_at=life['created_at'],app_stopped_at=life['stopped_at'],app_wallclock_seconds=seconds,app_wallclock_estimate_usd=seconds*RATE)
    else:job['app_wallclock_estimate_usd']=job['reservation_usd']  # unknown lifetime: keep the full reservation
    if out:
        job.update(measured_compute_usd=out['estimated_compute_usd'],elapsed_seconds=out['elapsed_seconds'],
                   complete=out['summary'].get('complete'),result_file=str(result))


if __name__=='__main__':main(*sys.argv[1:])
