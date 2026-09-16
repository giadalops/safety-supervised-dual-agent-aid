from __future__ import annotations

from typing import Any, Protocol
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .core import SafetySupervisor


class GlucoseSimulator(Protocol):
    """Adapter boundary for Simglucose 0.2.11 or another simulator.

    reset returns the initial CGM glucose (mg/dL). step receives insulin units
    for the 3-min interval and returns (next_glucose, terminated, info).
    """
    def reset(self, *, seed: int | None, subject: str, meals: Any = None) -> float: ...
    def step(self, insulin_u: float) -> tuple[float, bool, dict[str, Any]]: ...


class PrimaryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, simulator: GlucoseSimulator, supervisor: SafetySupervisor,
                 subject: str, scenario_seed: int = 1000):
        self.simulator, self.sup, self.subject = simulator, supervisor, subject
        self.scenario_seed = scenario_seed
        self.action_space = spaces.Discrete(self.sup.cfg.n_actions)
        self.observation_space = spaces.Box(0.0, 1000.0, shape=(1,), dtype=np.float32)
        self.g = self.prev_g = 112.0

    def action_masks(self) -> np.ndarray:
        return self.sup.action_mask(self.g, self.sup.slope(self.g, self.prev_g))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sup.reset()
        self.g = self.prev_g = float(self.simulator.reset(seed=seed or self.scenario_seed, subject=self.subject))
        return np.array([self.g], np.float32), {}

    def step(self, action):
        slope = self.sup.slope(self.g, self.prev_g)
        pred = self.sup.forecast(self.g, slope)
        u_cmd = self.sup.action_to_dose(int(action))
        u_floor, corrected = self.sup.predictive_floor(u_cmd, pred)
        next_g, terminated, info = self.simulator.step(u_floor)
        self.sup.record_delivery(u_floor, corrected)
        reward = self.sup.primary_reward(float(next_g))
        self.prev_g, self.g = self.g, float(next_g)
        return np.array([self.g], np.float32), reward, bool(terminated), False, info


class SafetyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, simulator: GlucoseSimulator, supervisor: SafetySupervisor,
                 frozen_primary, subject: str, scenario_seed: int = 2000):
        self.simulator, self.sup, self.primary = simulator, supervisor, frozen_primary
        self.subject, self.scenario_seed = subject, scenario_seed
        self.action_space = spaces.Discrete(len(self.sup.cfg.safety_factors))
        self.observation_space = spaces.Box(-1e4, 1e4, shape=(6,), dtype=np.float32)
        self.g = self.prev_g = 112.0
        self._features = np.zeros(6, np.float32)

    def _make_features(self) -> tuple[np.ndarray, bool]:
        slope = self.sup.slope(self.g, self.prev_g)
        pred = self.sup.forecast(self.g, slope)
        mask = self.sup.action_mask(self.g, slope)
        a, _ = self.primary.predict(np.array([self.g], np.float32), deterministic=True, action_masks=mask)
        floor, corrected = self.sup.predictive_floor(self.sup.action_to_dose(int(a)), pred)
        return np.array([self.g, slope, pred, floor, self.sup.iob_proxy(), self.sup.dose_30()], np.float32), corrected

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sup.reset()
        self.g = self.prev_g = float(self.simulator.reset(seed=seed or self.scenario_seed, subject=self.subject))
        self._features, _ = self._make_features()
        return self._features, {}

    def step(self, action):
        features, corrected = self._make_features()
        g, slope, pred, floor = map(float, features[:4])
        factor = self.sup.cfg.safety_factors[int(action)]
        final_u = self.sup.final_dose(factor * floor, g, slope, pred)
        next_g, terminated, info = self.simulator.step(final_u)
        reward = self.sup.safety_reward(float(next_g), pred, final_u, factor)
        self.sup.record_delivery(final_u, corrected)
        self.prev_g, self.g = self.g, float(next_g)
        self._features, _ = self._make_features()
        info = {**info, "u_final": final_u, "factor": factor}
        return self._features, reward, bool(terminated), False, info