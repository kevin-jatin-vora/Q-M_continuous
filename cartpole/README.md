# CartPole Lipschitz RA-DQN

Sibling project to `dollar_euro_16tile`: region-aware DQN pruning on
**Gymnasium CartPole-v1**, with regions **discovered from B1/B2 behavior data**.

## Setup

```bat
cd D:\dollar_euro_lipschitz_clean\cartpole
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## One-shot experiment

```bat
run_experiment.cmd --profile smoke
run_experiment.cmd --profile full --runs 5 --seed 0
```

Or:

```bat
.venv\Scripts\python.exe -u scripts\run_experiment.py --profile full --runs 5
```

## Pipeline stages

1. `scripts/collect_behaviors.py` — train B1/B2, dump transitions  
2. `scripts/discover_regions.py` — adaptive state tree (or grid)  
3. `scripts/compute_stats.py` — `(region, action)` means / radii / Lipschitz  
4. `scripts/train_q_single.py` — Q on mean dynamics  
5. `scripts/compute_stats.py` again — empirical `Lq` with Q_single  
6. `scripts/train_baseline_dqn.py` + `scripts/train_ra_dqn.py`  
7. `scripts/plot_returns.py`

## Docs

- [`algorithm.md`](algorithm.md) — behaviors, region discovery, stats, pruning math  
- [`RUN_ORDER.md`](RUN_ORDER.md) — stage-by-stage commands  
- [`ARTIFACTS.md`](ARTIFACTS.md) — output file map  

## Relation to 16-tile

| 16-tile | CartPole |
|---------|----------|
| Tile category from layout | Adaptive / grid partition of `s` |
| R1/R2 spatial peaks | B1/B2 angle-threshold behaviors |
| Combined Dollar–Euro reward | Classic CartPole +1/step |
| Stochastic σ noise | Deterministic env (`r` from binning) |
