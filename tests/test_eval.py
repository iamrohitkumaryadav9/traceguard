import numpy as np
import pytest

from src.ml.eval import (
    best_cost_threshold,
    best_f1_threshold,
    brier_score,
    cost_curve,
    expected_calibration_error,
    per_step_metrics,
    pr_auc,
    prf_at_threshold,
    roc_auc,
    topk_metrics,
    total_cost,
)

Y = np.array([1, 1, 0, 0, 0])
S = np.array([0.9, 0.4, 0.6, 0.2, 0.1])


def test_perfect_ranking():
    assert pr_auc([1, 0, 0], [0.9, 0.2, 0.1]) == pytest.approx(1.0)
    assert roc_auc([1, 0, 0], [0.9, 0.2, 0.1]) == pytest.approx(1.0)


def test_roc_auc_hand_made():
    # positives 0.9, 0.4 vs negatives 0.6, 0.2, 0.1 -> 5 of 6 pairs correctly ordered
    assert roc_auc(Y, S) == pytest.approx(5 / 6)


def test_prf_at_threshold():
    m = prf_at_threshold(Y, S, 0.5)  # predicted positive: 0.9, 0.6 -> tp=1 fp=1 fn=1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_prf_no_predictions():
    m = prf_at_threshold(Y, S, 2.0)
    assert m == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_best_f1_threshold():
    thr, f1 = best_f1_threshold(Y, S)
    assert thr == pytest.approx(0.4)  # predict 0.9, 0.6, 0.4 -> p=2/3, r=1, f1=0.8
    assert f1 == pytest.approx(0.8)


def test_per_step_metrics():
    t = np.array([1, 1, 1, 2, 2])
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
    out = per_step_metrics(t, y, s, 0.5).set_index("time_step")
    assert out.loc[1, "n_illicit"] == 2
    assert out.loc[1, "recall"] == pytest.approx(0.5)
    # ranking y,n,y -> AP = 1*(1/2) + (2/3)*(1/2)
    assert out.loc[1, "pr_auc"] == pytest.approx(5 / 6)
    assert out.loc[2, "n_illicit"] == 0
    assert np.isnan(out.loc[2, "recall"])
    assert np.isnan(out.loc[2, "pr_auc"])


def test_cost_curve_and_best_threshold():
    # C=5: flag nothing costs 10; thr 0.9 -> fn=1 -> 5; thr 0.6 -> fp=1,fn=1 -> 6; thr 0.4 -> fp=1 -> 1
    thr, cost = cost_curve(Y, S, c_miss=5)
    assert thr[0] == np.inf and cost[0] == 10
    assert dict(zip(thr[1:], cost[1:]))[0.9] == 5
    assert dict(zip(thr[1:], cost[1:]))[0.4] == 1
    t, c = best_cost_threshold(Y, S, c_miss=5)
    assert (t, c) == (0.4, 1)
    # top-scored item is a negative and misses are cheap (C=0.5): flagging nothing wins (cost 0.5)
    t, c = best_cost_threshold([0, 1], [0.9, 0.1], c_miss=0.5)
    assert t == np.inf and c == 0.5


def test_total_cost():
    assert total_cost(Y, S, 0.5, c_miss=10) == {"fp": 1, "fn": 1, "cost": 11}


def test_topk_metrics():
    m = topk_metrics(Y, S, 2)  # top 2: 0.9 (pos), 0.6 (neg)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["max_recall"] == pytest.approx(1.0)
    assert topk_metrics(Y, S, 1)["max_recall"] == pytest.approx(0.5)
    assert np.isnan(topk_metrics([0, 0], [0.1, 0.2], 1)["recall"])


def test_brier_and_ece():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
    # all predictions 0.8 and 80% positives -> perfectly calibrated
    assert expected_calibration_error([1, 1, 1, 1, 0], [0.8] * 5) == pytest.approx(0.0)
    # all predictions 0.9 but only half positive -> ECE 0.4
    assert expected_calibration_error([1, 0], [0.9, 0.9]) == pytest.approx(0.4)