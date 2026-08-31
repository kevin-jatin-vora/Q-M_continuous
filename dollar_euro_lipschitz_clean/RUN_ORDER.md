# Run Order

Run commands from the project root.

## 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Use the activated `.venv` for every command below. The base Anaconda
environment can load separate Intel OpenMP runtimes from NumPy and PyTorch,
causing `OMP: Error #15`. Do not use `KMP_DUPLICATE_LIB_OK`; it suppresses
the check without fixing the conflicting runtimes.

## 2. Complete experiment from CMD

`--sigma` overrides the config for every stage, including R1/R2 collection and
mean-CI radius regeneration. Heatmaps are written after center Q and before
DQN/RA-DQN. Full profile trains 5 agent seeds and plots
mean with a shaded 95% t CI. Empirical RA-DQN is on by default.

```bat
cd /d D:\dollar_euro_lipschitz_clean\dollar_euro_lipschitz_clean
run_experiment.cmd configs\default.json --sigma 0.00004 --profile full
```

Dry run first:

```bat
run_experiment.cmd configs\default.json --sigma 0.00004 --profile smoke --dry-run
```

Full sigma sweep (`0.00008` scaled by 1, 2, 3, 4, 5):

```bat
run_all_experiments.cmd
```

## 3. Train Center-Model Q

```bash
python scripts/train_q_single.py --iters 80000
```

For a quick sanity run:

```bash
python scripts/train_q_single.py --iters 1000
```

Using the bundled `data/single_q_bounds_from_json.json` requires
`--sigma 0.00008`. The experiment runner regenerates matching bounds first.

## 4. Train Baseline DQN

```bash
python scripts/train_baseline_dqn.py --steps 150000
```

For a quick sanity run:

```bash
python scripts/train_baseline_dqn.py --steps 2000 --eval-every 500
```

## 5. Train RA-DQN With Empirical Lq

```bash
python scripts/train_ra_dqn.py --lq-source empirical --steps 150000
```

Skip this step when only the theoretical comparison is needed. The same
sigma-matching rule as center-Q applies to the pruning JSON.

## 6. Train RA-DQN With Theoretical Lq

```bash
python scripts/train_ra_dqn.py --lq-source theoretical --steps 150000
```

## 7. Plot Results

```bash
python scripts/plot_returns.py outputs/dqn_runs.npy outputs/ra_dqn_empirical_runs.npy outputs/ra_dqn_theoretical_runs.npy --labels DQN RA-DQN-Empirical RA-DQN-Theoretical
```

Without an empirical run:

```bash
python scripts/plot_returns.py outputs/dqn_runs.npy outputs/ra_dqn_theoretical_runs.npy --labels DQN RA-DQN-Theoretical
```

## 8. Generate Heatmap and Tables

Once `q_single_region.pth` exists in an experiment directory:

```bash
python scripts/generate_analysis.py --lq-source theoretical --sigma 0.00008 --bounds outputs/experiments/<run>/transition_bounds.json --lipschitz outputs/experiments/<run>/lipschitz_constants.json --q-single outputs/experiments/<run>/q_single_region.pth --output-dir outputs/experiments/<run>
```

The explicit sigma matches the supplied precomputed transition bounds. The
project config default is `0.00004`; do not combine a different runtime sigma
with the fixed `0.00008` bounds unless the bounds have been re-estimated.
