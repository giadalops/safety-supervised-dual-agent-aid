import numpy as np
import pytest
from dual_agent_aid.core import SafetySupervisor


@pytest.fixture
def sup():
    return SafetySupervisor(40.0)


def test_action_grid(sup):
    assert sup.action_to_dose(0) == 0
    assert sup.action_to_dose(10) == pytest.approx(0.20)
    assert sup.action_to_dose(20) == pytest.approx(0.40)


def test_mask_priority_and_thresholds(sup):
    assert np.flatnonzero(sup.action_mask(80, -1)).tolist() == [0]
    assert np.flatnonzero(sup.action_mask(170, 0.3)).tolist() == list(range(10, 21))
    assert np.flatnonzero(sup.action_mask(210, 0.0)).tolist() == list(range(15, 21))


def test_predictive_floor_and_refractory(sup):
    floor, active = sup.predictive_floor(0.02, 240)
    assert active and floor == pytest.approx(0.32 * (240 - 112) / 40)
    sup.record_delivery(0.1, correction_active=True)
    floor2, active2 = sup.predictive_floor(0.02, 240)
    assert not active2 and floor2 == 0.02


def test_lgs_and_budget(sup):
    assert sup.final_dose(0.4, 85, -0.2, 120) == 0
    for _ in range(10):
        sup.record_delivery(0.08)
    assert sup.final_dose(0.4, 150, 0, 200) == pytest.approx(0.0)


def test_predictive_cap(sup):
    # (102 - 98) / 40 = 0.1 U
    assert sup.final_dose(0.4, 120, 0, 102) == pytest.approx(0.1)