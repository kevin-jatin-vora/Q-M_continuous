# Artifacts

## Included Inputs

| File | Meaning |
|---|---|
| `data/single_q_bounds_from_json.json` | Region-action mean transitions and Student-t confidence radii |
| `data/lipschitz_constants_from_doc.json` | Lr, empirical Lq, and theoretical Lq constants from the writeup |
| `docs/lipschitz_pruning_dollar_euro.docx` | Original technical writeup |
| `original_uploads/` | Untouched uploaded files |

## Generated Outputs

| File | Created By |
|---|---|
| `outputs/q_single_region.pth` | `scripts/train_q_single.py` |
| `outputs/dqn.pth` | `scripts/train_baseline_dqn.py` |
| `outputs/dqn.npy` | `scripts/train_baseline_dqn.py` |
| `outputs/ra_dqn_empirical.pth` | `scripts/train_ra_dqn.py --lq-source empirical` |
| `outputs/ra_dqn_empirical.npy` | `scripts/train_ra_dqn.py --lq-source empirical` |
| `outputs/ra_dqn_theoretical.pth` | `scripts/train_ra_dqn.py --lq-source theoretical` |
| `outputs/ra_dqn_theoretical.npy` | `scripts/train_ra_dqn.py --lq-source theoretical` |
| `outputs/returns_plot.png` | `scripts/plot_returns.py` |

## Missing From Uploads

The original scripts referenced these files, but they were not included in the uploaded set:

| Missing File | Clean Replacement |
|---|---|
| `Q_single_region.pth` | Generate `outputs/q_single_region.pth` with `scripts/train_q_single.py` |
| `two_reward_dqn_outputs/two_reward_dqn_summary.json` | Replaced for pruning by `data/lipschitz_constants_from_doc.json` |
| `offline_transitions.npz` | Original-only visualization dependency; not required for cleaned scripts |
| `two_reward_dqn_raw_transitions.pkl` | Original model-based dependency; not required for cleaned scripts |
