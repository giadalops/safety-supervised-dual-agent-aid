from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass(frozen=True)
class ControllerConfig:
    dt_min: float = 3.0
    target: float = 112.0
    n_actions: int = 21
    primary_max_u: float = 0.40
    s1: float = 220.0
    s2: float = 250.0
    kp_lo: float = 0.32
    kp_hi: float = 0.52
    min_correction_u: float = 0.02
    refractory_min: float = 20.0
    max_30min_u: float = 0.8
    g_pred: float = 98.0
    safety_factors: tuple[float, ...] = tuple(0.50 + 0.05 * k for k in range(15))
    iob_horizon_min: float = 240.0

    @property
    def action_delta(self) -> float:
        return self.primary_max_u / (self.n_actions - 1)

    @property
    def dose_window_steps(self) -> int:
        return round(30 / self.dt_min)

    @property
    def iob_steps(self) -> int:
        return round(self.iob_horizon_min / self.dt_min)


@dataclass
class ControllerState:
    delivered: deque[float] = field(default_factory=deque)
    minutes_since_correction: float = float("inf")


class SafetySupervisor:
    """Pure implementation of the paper's equations, independent of simulator."""

    def __init__(self, cf_mgdl_per_u: float, cfg: ControllerConfig | None = None):
        if cf_mgdl_per_u <= 0:
            raise ValueError("CF must be positive")
        self.cf = float(cf_mgdl_per_u)
        self.cfg = cfg or ControllerConfig()
        self.state = ControllerState(deque(maxlen=self.cfg.iob_steps))

    def reset(self) -> None:
        self.state = ControllerState(deque(maxlen=self.cfg.iob_steps))

    def slope(self, glucose: float, previous_glucose: float) -> float:
        return (glucose - previous_glucose) / self.cfg.dt_min

    @staticmethod
    def forecast(glucose: float, slope: float) -> float:
        return glucose + 30.0 * slope

    def action_to_dose(self, action: int) -> float:
        if not 0 <= int(action) < self.cfg.n_actions:
            raise ValueError("invalid primary action")
        return int(action) * self.cfg.action_delta

    def action_mask(self, glucose: float, slope: float) -> np.ndarray:
        valid = np.ones(self.cfg.n_actions, dtype=bool)
        if glucose < 90 or (glucose < 120 and slope < -0.5):
            valid[:] = False
            valid[0] = True
        elif glucose > 160 and slope > 0.2:
            valid[:10] = False
        elif glucose > 200:
            valid[:15] = False
        return valid

    def primary_reward(self, glucose: float) -> float:
        r = 1.0 - (abs(glucose - self.cfg.target) / 60.0) ** 2
        if glucose > 180:
            r -= 0.5 + 0.003 * (glucose - 180)
        if glucose < 70:
            r -= 1.5 + 0.01 * (70 - glucose)
        if glucose < 50 or glucose > 360:
            r -= 8.0
        return float(r)

    def dose_30(self) -> float:
        return float(sum(list(self.state.delivered)[-self.cfg.dose_window_steps :]))

    def iob_proxy(self) -> float:
        doses = np.asarray(self.state.delivered, dtype=float)
        if doses.size == 0:
            return 0.0
        weights_old_to_new = np.linspace(0.0, 1.0, doses.size, endpoint=True)
        return float(doses @ weights_old_to_new)

    def predictive_floor(self, u_cmd: float, predicted_glucose: float) -> tuple[float, bool]:
        eligible = self.state.minutes_since_correction >= self.cfg.refractory_min
        if predicted_glucose <= self.cfg.s1 or not eligible:
            return float(u_cmd), False
        kp = self.cfg.kp_lo if predicted_glucose <= self.cfg.s2 else self.cfg.kp_hi
        error = max(0.0, predicted_glucose - self.cfg.target)
        u_corr = max(self.cfg.min_correction_u, kp * error / self.cf)
        return max(float(u_cmd), u_corr), True

    @staticmethod
    def low_glucose_suspend(glucose: float, slope: float) -> bool:
        return (glucose < 90 and slope <= -0.1) or (glucose < 100 and slope <= -0.3)

    def final_dose(self, u_mod: float, glucose: float, slope: float, predicted_glucose: float) -> float:
        if self.low_glucose_suspend(glucose, slope):
            return 0.0
        predictive_cap = max(0.0, (predicted_glucose - self.cfg.g_pred) / self.cf)
        residual_budget = max(0.0, self.cfg.max_30min_u - self.dose_30())
        return float(min(max(0.0, u_mod), predictive_cap, residual_budget))

    def safety_reward(self, next_glucose: float, predicted_glucose: float,
                      final_u: float, factor: float,
                      lambda_pred: float = 1.0, lambda_ins: float = 0.12,
                      lambda_dev: float = 0.02) -> float:
        r = 0.5 * (70 <= next_glucose <= 180) + 1 - 0.002 * abs(next_glucose - 112)
        r -= 0.005 * max(0.0, next_glucose - 180)
        r -= 0.02 * max(0.0, 70 - next_glucose)
        r -= lambda_pred * max(0.0, self.cfg.g_pred - predicted_glucose) / 30.0
        r -= lambda_ins * final_u + lambda_dev * abs(factor - 1.0)
        if next_glucose < 50 or next_glucose > 360:
            r -= 8.0
        return float(r)

    def record_delivery(self, dose: float, correction_active: bool = False) -> None:
        self.state.delivered.append(float(dose))
        if correction_active:
            self.state.minutes_since_correction = 0.0
        else:
            self.state.minutes_since_correction += self.cfg.dt_min