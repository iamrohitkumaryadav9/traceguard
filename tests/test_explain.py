import numpy as np
import pytest

from src.ml.explain import feature_group, shap_values_positive


def test_feature_groups():
    assert feature_group("f1") == "local"
    assert feature_group("f93") == "local"
    assert feature_group("f94") == "aggregated"
    assert feature_group("f165") == "aggregated"
    assert feature_group("chain_len") == "structural"
    with pytest.raises(ValueError):
        feature_group("mystery")


def test_shap_output_shapes():
    a = np.ones((4, 3))
    assert shap_values_positive(a).shape == (4, 3)
    assert shap_values_positive([np.zeros((4, 3)), a]).shape == (4, 3)
    assert shap_values_positive(np.stack([np.zeros((4, 3)), a], axis=-1)).shape == (4, 3)
