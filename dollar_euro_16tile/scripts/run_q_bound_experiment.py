"""Reuse an existing experiment's data, train learned Q bounds, analyze them, then train RA-DQN with those frozen bounds."""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(name,cmd):
    print('\n=== '+name+' ===\n'+subprocess.list2cmdline([str(x) for x in cmd]),flush=True)
    r=subprocess.run([str(x) for x in cmd],cwd=ROOT)
    if r.returncode: raise SystemExit(r.returncode)
def find(exp,name):
    for root in (exp/'data',exp/'models',exp):
        p=root/name
        if p.is_file(): return p
    raise FileNotFoundError(name)
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--experiment-dir',required=True)
    p.add_argument('--lq-source',choices=['empirical','theoretical'],default='theoretical'); p.add_argument('--iters',type=int,default=80000)
    p.add_argument('--steps',type=int,default=300000); p.add_argument('--eval-every',type=int,default=20000); p.add_argument('--eval-episodes',type=int,default=30)
    p.add_argument('--seed',type=int,default=0); p.add_argument('--stop-after-bounds',action='store_true'); a=p.parse_args()
    exp=Path(a.experiment_dir).resolve(); rc=find(exp,'run_config.json'); meta=json.loads(rc.read_text(encoding='utf-8'))
    config=Path(meta.get('config_path',ROOT/'configs'/'default.json')); sigma=float(meta['effective_sigma']); det=float(meta['effective_determinism'])
    gamma=float(meta.get('effective_gamma',meta.get('training',{}).get('gamma',0.99))); ds=float(meta.get('effective_deterministic_sigma_scale',0.0))
    bounds=find(exp,'transition_bounds.json'); lips=find(exp,'lipschitz_constants.json'); py=sys.executable
    models=exp/'models'; plots=exp/'plots'/f'learned_q_bounds_{a.lq_source}'; models.mkdir(exist_ok=True); plots.mkdir(parents=True,exist_ok=True)
    qu=models/f'q_ub_{a.lq_source}.pth'; ql=models/f'q_lb_{a.lq_source}.pth'
    env=['--config',config,'--sigma',sigma,'--gamma',gamma,'--determinism',det,'--deterministic-sigma-scale',ds]
    run('1/3 Train learned Q upper/lower bounds',[py,'scripts/train_q_bounds.py',*env,'--bounds',bounds,'--lipschitz',lips,'--lq-source',a.lq_source,'--iters',a.iters,'--seed',a.seed,'--out-ub',qu,'--out-lb',ql])
    run('2/3 Analyze learned Q bounds',[py,'scripts/generate_q_bound_analysis.py','--q-ub',qu,'--q-lb',ql,'--output-dir',plots])
    if a.stop_after_bounds: return
    prefix=models/f'ra_dqn_learned_bounds_{a.lq_source}'
    run('3/3 Train RA-DQN using learned Q bounds',[py,'scripts/train_ra_dqn_bounds.py',*env,'--q-ub',qu,'--q-lb',ql,'--lq-source',a.lq_source,'--steps',a.steps,'--eval-every',a.eval_every,'--eval-episodes',a.eval_episodes,'--seed',a.seed,'--out-prefix',prefix])
if __name__=='__main__': main()
