import numpy as np
import pytest

from src.ml.eval import best_f1_threshold, per_step_metrics, pr_auc, prf_at_threshold, roc_auc

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
