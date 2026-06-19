import math

import numpy as np
import pandas as pd
import pytest

from src.models.model_utils import hhmm_to_hour, prepare_sequences
from src.routing.a_star import a_star_search, get_path_cost, yen_k_shortest_paths
from src.routing.graph import RoadNetworkGraph
from src.routing.speed_converter import calculate_travel_time, flow_to_speed


def build_test_graph():
    graph = RoadNetworkGraph()
    for node_id in range(1, 5):
        graph.add_node(node_id, 0.0, 0.0)

    graph.add_edge(1, 2, 1.0)
    graph.add_edge(2, 4, 1.0)
    graph.add_edge(1, 3, 1.0)
    graph.add_edge(3, 4, 2.0)
    graph.add_edge(1, 4, 10.0)
    return graph


def test_flow_to_speed_caps_low_flow_at_speed_limit():
    assert flow_to_speed(351.0) == 60.0
    assert flow_to_speed(100.0) == 60.0


def test_flow_to_speed_uses_capacity_speed_when_flow_exceeds_quadratic_capacity():
    assert flow_to_speed(2000.0) == 32.0


def test_calculate_travel_time_adds_intersection_delay_and_handles_zero_speed():
    assert calculate_travel_time(60.0, 1.0, is_intersection=True) == 90.0

    expected = (1.6 / 32.0) * 3600.0
    assert calculate_travel_time(0.0, 1.6) == expected


def test_a_star_returns_lowest_cost_path():
    graph = build_test_graph()

    path, cost = a_star_search(graph, 1, 4)

    assert path == [1, 2, 4]
    assert cost == 2.0


def test_a_star_respects_ignored_edges():
    graph = build_test_graph()

    path, cost = a_star_search(graph, 1, 4, ignored_edges={(1, 2)})

    assert path == [1, 3, 4]
    assert cost == 3.0


def test_get_path_cost_returns_infinity_for_missing_edge():
    graph = build_test_graph()

    assert math.isinf(get_path_cost(graph, [2, 3]))


def test_yen_k_shortest_paths_returns_sorted_unique_paths():
    graph = build_test_graph()

    paths = yen_k_shortest_paths(graph, 1, 4, k=3)

    assert paths == [
        ([1, 2, 4], 2.0),
        ([1, 3, 4], 3.0),
        ([1, 4], 10.0),
    ]


def test_load_from_csv_builds_nodes_and_prediction_weighted_edges(tmp_path):
    lookup_path = tmp_path / "movement_lookup.csv"
    edges_path = tmp_path / "edges.csv"

    pd.DataFrame(
        [
            {"scat_num": 100, "latitude": 0.0, "longitude": 0.0},
            {"scat_num": 100, "latitude": -37.80, "longitude": 145.00},
            {"scat_num": 200, "latitude": -37.82, "longitude": 145.02},
        ]
    ).to_csv(lookup_path, index=False)
    pd.DataFrame(
        [
            {
                "from_site": 100,
                "to_site": 200,
                "travel_distance_km": 1.0,
            }
        ]
    ).to_csv(edges_path, index=False)

    class FixedFlowModel:
        def predict(self, from_site, to_site, time_str):
            assert (from_site, to_site, time_str) == (100, 200, "1100")
            return 351.0

    graph = RoadNetworkGraph()
    graph.load_from_csv(str(lookup_path), str(edges_path), FixedFlowModel(), "1100")

    assert graph.nodes[100].lat == -37.80
    assert graph.nodes[100].lng == 145.00
    assert graph.nodes[100].neighbors == [(200, 90.0)]
    assert graph.nodes[200].neighbors == [(100, 90.0)]


def test_hhmm_to_hour_accepts_valid_inputs_and_rejects_invalid_times():
    assert hhmm_to_hour("0830") == 8
    assert hhmm_to_hour(930) == 9

    with pytest.raises(ValueError, match="Hour must be between"):
        hhmm_to_hour("2460")

    with pytest.raises(ValueError, match="Minute must be between"):
        hhmm_to_hour("1260")


def test_prepare_sequences_keeps_movements_separate_and_splits_chronologically():
    rows = []
    for movement_id, scats_number in [("A", 100), ("B", 200)]:
        for hour in range(6):
            rows.append(
                {
                    "movement_id": movement_id,
                    "scats_number": scats_number,
                    "DateTime": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour),
                    "dayofweek": hour % 7,
                    "isweekend": 0,
                    "hourly_traffic_volume": 100 + hour,
                    "hour": hour,
                }
            )
    df = pd.DataFrame(rows)

    X_train, y_train, X_test, y_test, scaler, cutoff, test_dates, test_movements = (
        prepare_sequences(df, window_size=2, train_ratio=0.5)
    )

    assert X_train.shape == (2, 2, 3)
    assert y_train.shape == (2,)
    assert X_test.shape == (6, 2, 3)
    assert y_test.shape == (6,)
    assert set(test_movements) == {"A", "B"}
    assert np.datetime64(cutoff) == np.datetime64("2026-01-01T03:00:00")
    assert all(np.datetime64(date) >= np.datetime64(cutoff) for date in test_dates)
    assert scaler.data_min_[0] == 100.0
    assert scaler.data_max_[0] == 102.0
