from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import yaml


def _load_factory(spec: str):
    module, function = spec.split(":", 1)
    return getattr(importlib.import_module(module), function)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sim-factory", required=True, help="module:function")
    p.add_argument("--subject", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    from stable_baselines3 import PPO
    from sb3_contrib import MaskablePPO
    from .core import ControllerConfig, SafetySupervisor
    from .envs import PrimaryEnv, SafetyEnv

    cfg = yaml.safe_load(Path(args.config).read_text())
    subject = next(s for s in cfg["subjects"] if s["name"] == args.subject)
    factory = _load_factory(args.sim_factory)
    cc = ControllerConfig(iob_horizon_min=cfg["iob_proxy"]["horizon_minutes"])
    policy_kwargs = {"net_arch": [64, 64]}
    
    p1 = cfg["training"]["primary"]
    env1 = PrimaryEnv(factory(cfg), SafetySupervisor(subject["cf_mgdl_per_u"], cc), args.subject)
    primary = MaskablePPO("MlpPolicy", env1, seed=args.seed, policy_kwargs=policy_kwargs,
                          n_steps=p1["n_steps"], n_epochs=p1["n_epochs"], batch_size=p1["batch_size"],
                          learning_rate=p1["learning_rate"], gamma=p1["gamma"],
                          gae_lambda=p1["gae_lambda"], clip_range=p1["clip_range"], verbose=1)
    primary.learn(total_timesteps=p1["episodes"] * cfg["paper"]["steps_per_episode"])
    args.output.mkdir(parents=True, exist_ok=True)
    primary.save(args.output / "primary")

    p2 = cfg["training"]["safety"]
    env2 = SafetyEnv(factory(cfg), SafetySupervisor(subject["cf_mgdl_per_u"], cc), primary, args.subject)
    safety = PPO("MlpPolicy", env2, seed=args.seed, policy_kwargs=policy_kwargs,
                 n_steps=p2["n_steps"], n_epochs=p2["n_epochs"], batch_size=p2["batch_size"],
                 learning_rate=p2["learning_rate"], gamma=p2["gamma"],
                 gae_lambda=p2["gae_lambda"], clip_range=p2["clip_range"], verbose=1)
    safety.learn(total_timesteps=p2["episodes"] * cfg["paper"]["steps_per_episode"])
    safety.save(args.output / "safety_adjuster")


if __name__ == "__main__":
    main()