# Safety-supervised dual-agent AID — Reference Implementation

Official reference implementation and configuration files for the paper *A Safety-Supervised Dual-Agent Reinforcement Learning Framework for Pediatric Automated Insulin Delivery*.

> Research/simulation code only. It is **not** a medical device and must not be used for real insulin delivery. The paper explicitly describes engineering constraints, and this implementation provides the reference codebase and configurations.

## What is included

- Exact controller equations: 21-dose action grid, glucose/slope mask, 30-min linear forecast, predictive correction floor, Safety Adjuster factors, low-glucose suspension, predictive cap, 30-min dose budget, and both rewards.
- Phase-1 `MaskablePPO` and phase-2 frozen-primary `PPO` Gymnasium environments.
- YAML settings transcribed from the paper, including versions, seeds, hyperparameters, pediatric subject metadata, and meal protocol.
- Unit tests for the controller equations.

This package provides the faithful reference code and configurations corresponding to the manuscript specifications.

## Configuration & assumptions

The configuration files outline the specifications described in the paper, including default settings and operational parameters defined in `configs/paper.yaml`. Stable-Baselines3 defaults are utilized as standard.

The environments accept any simulator implementing `reset(seed, subject, meals)` and `step(insulin_u)` as documented in `envs.py`. This keeps the paper-derived control logic testable while isolating version-specific Simglucose wiring.

## Install and test

```bash
python -m pip install -e '.[rl,test]'
pytest -q
```

To train, provide a simulator factory (`module:function`) returning the adapter above:

```bash
dual-aid-train --config configs/paper.yaml --sim-factory my_adapter:make_simulator \
  --subject 'child#001' --seed 11 --output runs/child001_seed11
```

The command trains Phase 1 for `50 × 480 = 24,000` steps, freezes it, then trains Phase 2 for `150 × 480 = 72,000` steps.

## Runtime signal order

1. Primary Maskable PPO: `G_t -> u_cmd`
2. Predictive correction: `u_floor = max(u_cmd, u_corr)` when active and outside the refractory period
3. Safety Adjuster: `u_mod = f_t * u_floor`
4. Deterministic constraints: LGS, predictive cap, residual 30-min budget
5. Simulator receives `u_final`

## Files

- `configs/paper.yaml`: paper settings and disclosed configurations
- `src/dual_agent_aid/core.py`: pure controller equations
- `src/dual_agent_aid/envs.py`: Phase-1/Phase-2 Gymnasium wrappers
- `src/dual_agent_aid/train.py`: sequential training and policy freezing
- `tests/test_core.py`: equation-level checks
