"""Plot learned Q_UB/Q_LB diagnostics and learned-bound pruning heatmap."""
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np, torch
import matplotlib.pyplot as plt
from dollar_euro_lipschitz.models import QNet
from dollar_euro_lipschitz.q_bounds import load_frozen_qnet, learned_bound_allowed_mask

def save_img(z, path, title, label, grid):
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(z,origin='lower',extent=[0,1,0,1],aspect='equal')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(title); fig.colorbar(im,ax=ax,label=label)
    fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches='tight'); plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--q-ub',required=True); p.add_argument('--q-lb',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--grid-size',type=int,default=201); p.add_argument('--tol',type=float,default=1e-5)
    a=p.parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ub=load_frozen_qnet(a.q_ub,QNet,dev); lb=load_frozen_qnet(a.q_lb,QNet,dev)
    c=np.linspace(0,1,a.grid_size,dtype=np.float32); xx,yy=np.meshgrid(c,c); s=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32)
    with torch.inference_mode():
        st=torch.as_tensor(s,device=dev); qu=ub(st); ql=lb(st); mask=learned_bound_allowed_mask(qu,ql,a.tol)
        vu=qu.max(1).values.cpu().numpy(); vl=ql.max(1).values.cpu().numpy(); width=(qu-ql).mean(1).cpu().numpy()
        violations=(ql>qu).sum(1).cpu().numpy(); surviving=mask.sum(1).cpu().numpy()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); shape=(a.grid_size,a.grid_size)
    save_img(vu.reshape(shape),out/'q_ub_value_heatmap.png','Learned upper bound: max_a Q_UB','max_a Q_UB',a.grid_size)
    save_img(vl.reshape(shape),out/'q_lb_value_heatmap.png','Learned lower bound: max_a Q_LB','max_a Q_LB',a.grid_size)
    save_img(width.reshape(shape),out/'q_bound_width_heatmap.png','Mean action-wise Q_UB - Q_LB','mean width',a.grid_size)
    save_img(violations.reshape(shape),out/'q_bound_violation_heatmap.png','Ordering violations (diagnostic only)','count Q_LB > Q_UB',a.grid_size)
    save_img(surviving.reshape(shape),out/'pruning_heatmap_learned_bounds.png','Actions surviving learned Q bounds','actions',a.grid_size)
    total=qu.numel(); bad=int((ql>qu).sum().item()); print(f'ordering violations: {bad}/{total} ({100*bad/total:.6f}%)')
    print(f'min width={float((qu-ql).min()):.6g} mean width={float((qu-ql).mean()):.6g}')
    print(f'saved diagnostics to {out.resolve()}')
if __name__=='__main__': main()
