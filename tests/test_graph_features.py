import pandas as pd

from src.ml.graph_features import compute_graph_features, edges_cross_time_steps

# fan-in hub:  1,2,3 -> 4
# fan-out hub: 5 -> 6,7,8
# chain:       10 -> 11 -> 12 -> 13 -> 14
EDGES = pd.DataFrame(
    {
        "txId1": [1, 2, 3, 5, 5, 5, 10, 11, 12, 13],
        "txId2": [4, 4, 4, 6, 7, 8, 11, 12, 13, 14],
    }
)
IDS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]


def feats():
    return compute_graph_features(IDS, EDGES).set_index(pd.Index(IDS))


def test_degrees():
    f = feats()
    assert f.loc[4, "in_deg"] == 3 and f.loc[4, "out_deg"] == 0
    assert f.loc[5, "out_deg"] == 3 and f.loc[5, "in_deg"] == 0
    assert f.loc[12, "in_deg"] == 1 and f.loc[12, "out_deg"] == 1


def test_two_hop_counts():
    f = feats()
    assert f.loc[5, "fwd_2hop"] == 3  # 6, 7, 8
    assert f.loc[4, "bwd_2hop"] == 3  # 1, 2, 3
    assert f.loc[10, "fwd_2hop"] == 2  # 11, 12
    assert f.loc[14, "bwd_2hop"] == 2  # 13, 12
    assert f.loc[10, "bwd_2hop"] == 0


def test_component_size():
    f = feats()
    assert (f.loc[[1, 2, 3, 4], "wcc_size"] == 4).all()
    assert (f.loc[[5, 6, 7, 8], "wcc_size"] == 4).all()
    assert (f.loc[[10, 11, 12, 13, 14], "wcc_size"] == 5).all()


def test_neighbor_max_degree():
    f = feats()
    assert f.loc[1, "nbr_max_in_deg"] == 3  # neighbor 4 has in_deg 3
    assert f.loc[6, "nbr_max_out_deg"] == 3  # neighbor 5 has out_deg 3
    assert f.loc[12, "nbr_max_in_deg"] == 1


def test_chain_len():
    f = feats()
    # 10 has in_deg 0 and 14 has out_deg 0, so they are not links; 11, 12, 13 form the chain
    assert f.loc[[10, 14], "chain_len"].tolist() == [0, 0]
    assert f.loc[[11, 12, 13], "chain_len"].tolist() == [3, 3, 3]
    # hubs and leaves are not links
    assert (f.loc[[1, 2, 3, 4, 5, 6, 7, 8], "chain_len"] == 0).all()


def test_chain_stops_at_branch():
    # 20 -> 21 -> 22 -> (23, 24): 22 has out_deg 2 so it is a link but cannot be walked through forward
    edges = pd.DataFrame({"txId1": [19, 20, 21, 22, 22], "txId2": [20, 21, 22, 23, 24]})
    ids = [19, 20, 21, 22, 23, 24]
    f = compute_graph_features(ids, edges).set_index(pd.Index(ids))
    assert f.loc[[20, 21, 22], "chain_len"].tolist() == [3, 3, 3]


def test_chain_cap():
    n = 120
    ids = list(range(n))
    edges = pd.DataFrame({"txId1": ids[:-1], "txId2": ids[1:]})
    f = compute_graph_features(ids, edges)
    assert f["chain_len"].max() == 50


def test_edges_cross_time_steps():
    nodes = pd.DataFrame({"txId": [1, 2, 3], "time_step": [1, 1, 2]})
    assert edges_cross_time_steps(nodes, pd.DataFrame({"txId1": [1], "txId2": [2]})) == 0
    assert edges_cross_time_steps(nodes, pd.DataFrame({"txId1": [1], "txId2": [3]})) == 1
