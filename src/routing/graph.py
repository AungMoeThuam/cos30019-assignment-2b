# src/routing/graph.py
# This module defines the Graph data structure used to model the street network of Boroondara.

class RoadNetworkGraph:
    # TODO:
    # 1. Define the __init__ method:
    #    - Initialize containers for nodes (intersections) and edges (road segments).
    #    - Store coordinates (latitude and longitude) for each node (useful for A* heuristic).
    
    def __init__(self):
        self.nodes = {}  # {node_id: (lat, lon)}
        self.edges = {}  # {node_id: {neighbor_id: {'distance': dist, 'weight': travel_time}}}
        pass

    # TODO:
    # 1. Define method to add intersections:
    #    - `add_node(self, node_id, lat, lon)`
    
    def add_node(self, node_id, lat, lon):
        pass

    # TODO:
    # 1. Define method to add road connections:
    #    - `add_edge(self, u, v, distance)`
    
    def add_edge(self, u, v, distance):
        pass

    # TODO:
    # 1. Define method to dynamically update edge costs:
    #    - `update_edge_weights(self, predicted_flows)`
    #    - For each edge, take the predicted flow, call the speed converter to get travel time,
    #      and update the edge weight in `self.edges`.
    
    def update_edge_weights(self, predicted_flows):
        pass

if __name__ == "__main__":
    # Test network graph creation
    pass
