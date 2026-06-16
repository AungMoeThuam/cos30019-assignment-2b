# src/routing/graph.py
# This module defines the Graph data structure used to model the street network of Boroondara.

from dataclasses import dataclass, field
import pandas as pd
from src.routing.speed_converter import flow_to_speed, calculate_travel_time


@dataclass
class Node:
    """
    Represents a single node in the route-finding graph.

    :param id: unique node identifier
    :param lat: latitude coordinate
    :param lng: longitude coordinate
    :param neighbors: list of (neighbor_node_id, cost) tuples
    """

    id: int
    lat: float
    lng: float
    neighbors: list[tuple[int, float]] = field(default_factory=list)


class RoadNetworkGraph:
    def __init__(self):
        self.nodes = {}  # {node_id: Node}

    def add_node(self, node_id: int, lat: float, lng: float):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(id=node_id, lat=lat, lng=lng)

    def add_edge(self, u: int, v: int, cost: float):
        if u in self.nodes and v in self.nodes:
            # Bidirectional edge
            self._add_neighbor(u, v, cost)
            self._add_neighbor(v, u, cost)

    def _add_neighbor(self, node_id: int, neighbor_id: int, cost: float):
        node = self.nodes[node_id]
        # Remove existing edge if any
        node.neighbors = [n for n in node.neighbors if n[0] != neighbor_id]
        node.neighbors.append((neighbor_id, cost))

    def load_from_csv(
        self, lookup_path: str, edges_path: str, model_instance, time_str: str
    ):
        """
        Load node coordinates and connection edges from CSVs.
        Computes node coordinates by averaging non-zero movement sensor coordinates.
        Edge weights (costs) are dynamically computed using model predictions at departure time.
        """
        # 1. Compute node coordinates from lookup
        df_lookup = pd.read_csv(lookup_path)
        for scat_num, group in df_lookup.groupby("scat_num"):
            # Filter out zero coordinates
            valid = group[(group["latitude"] != 0) & (group["longitude"] != 0)]
            if len(valid) > 0:
                lat = valid["latitude"].mean()
                lng = valid["longitude"].mean()
            else:
                lat = group["latitude"].mean()
                lng = group["longitude"].mean()
            self.add_node(int(scat_num), float(lat), float(lng))

        # 2. Add edges using model predictions
        df_edges = pd.read_csv(edges_path)
        df_edges = df_edges.dropna(subset=["from_site", "to_site"])

        for _, row in df_edges.iterrows():
            u = int(row["from_site"])
            v = int(row["to_site"])
            dist = float(row["travel_distance_km"])

            # Predict flow using dependency-injected model
            flow = model_instance.predict(from_site=u, to_site=v, time_str=time_str)

            # Convert to speed and then travel time
            speed = flow_to_speed(flow)
            travel_time = calculate_travel_time(speed, dist, is_intersection=True)

            # Cost in graph is the travel time (in seconds)
            self.add_edge(u, v, travel_time)
