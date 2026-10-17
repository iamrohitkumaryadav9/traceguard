import pytest

from src.data import load_elliptic
from src.split import assign_split


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "elliptic_txs_classes.csv").write_text(
        "txId,class\n10,1\n20,2\n30,unknown\n40,2\n"
    )
    (tmp_path / "elliptic_txs_edgelist.csv").write_text("txId1,txId2\n10,20\n20,30\n")
    (tmp_path / "elliptic_txs_features.csv").write_text(
        "10,1,0.5,1.5\n20,1,0.1,0.2\n30,2,0.3,0.4\n40,40,0.9,0.8\n"
    )
    return tmp_path


def test_headerless_features_named(raw_dir):
    nodes, edges = load_elliptic(raw_dir)
    assert list(nodes.columns) == ["txId", "time_step", "f1", "f2", "label"]
    assert len(nodes) == 4
    assert nodes.loc[nodes.txId == 10, "f2"].item() == pytest.approx(1.5)
    assert list(edges.columns) == ["txId1", "txId2"]
    assert len(edges) == 2


def test_label_mapping(raw_dir):
    nodes, _ = load_elliptic(raw_dir)
    labels = dict(zip(nodes["txId"], nodes["label"]))
    assert labels == {10: "illicit", 20: "licit", 30: "unknown", 40: "licit"}


def test_unexpected_class_code_raises(raw_dir):
    (raw_dir / "elliptic_txs_classes.csv").write_text("txId,class\n10,3\n")
    with pytest.raises(ValueError):
        load_elliptic(raw_dir)


def test_split_assignment(raw_dir):
    nodes, _ = load_elliptic(raw_dir)
    split = dict(zip(assign_split(nodes)["txId"], assign_split(nodes)["split"]))
    assert split == {10: "train", 20: "train", 30: "unlabeled", 40: "test"}
