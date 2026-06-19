# A2B.py
# This is the main Command Line Interface (CLI) entry point for the TBRGS application.
# Positional arguments format:
#     python A2B.py <origin_node> <destination_node> <time> <model_type>
# Example:
#     python A2B.py 2000 2825 1100 LSTM

import sys
import os
from src.models.registry import get_model
from src.routing.graph import RoadNetworkGraph
from src.routing.a_star import a_star_search


def run_routing_and_prediction(
    origin_node: int,
    dest_node: int,
    time_str: str,
    model_name: str,
    map_file_path: str = "map.txt",
) -> bool:
    # Validate model selection via registry
    try:
        model_class = get_model(model_name)
    except ValueError as e:
        print(f"Error: {e}")
        return False

    # Paths to CSV data
    lookup_path = "data/processed/movement_lookup.csv"
    edges_path = "data/processed/edges.csv"

    if not os.path.exists(lookup_path) or not os.path.exists(edges_path):
        print(f"Error: Data files not found. Ensure '{
              lookup_path}' and '{edges_path}' exist.")
        return False

    # 3. Load Model (Dependency Injection)
    try:
        model_instance = model_class()
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    hour = int(time_str[:2])
    minute = int(time_str[2:])
    print(f"Welcome to TBRGS (Traffic-Based Route Guidance System)!")
    print(f"Loading road network graph and predicting traffic flows using {
          model_name} for departure time {hour:02d}:{minute:02d}...")

    # 4. Build Map Graph
    graph = RoadNetworkGraph()
    graph.load_from_csv(lookup_path, edges_path, model_instance, time_str)

    # Verify origin/destination node exist in graph
    if origin_node not in graph.nodes:
        print(f"Error: Origin node {
              origin_node} not found in the road network.")
        return False
    if dest_node not in graph.nodes:
        print(f"Error: Destination node {
              dest_node} not found in the road network.")
        return False

    # 5. Run A* Search
    path, travel_time = a_star_search(graph, origin_node, dest_node)

    # 6. Print Results
    if path:
        mins = int(travel_time // 60)
        secs = int(travel_time % 60)
        print(f"\nRouting from {origin_node} to {dest_node} succeeded!")
        print(f"Fastest Route: {' -> '.join(map(str, path))}")
        print(f"Estimated Travel Time: {mins} min {
              secs} s ({travel_time:.2f} seconds)\n")
    else:
        print(f"\nRouting failed: No path exists between {
              origin_node} and {dest_node}.\n")

    # 7. Write map.txt (which will also be overwritten during interactions in the visualizer)
    try:
        with open(map_file_path, "w") as f:
            f.write(f"{origin_node}\n")
            f.write(f"{dest_node}\n")
            # Write all nodes sorted by ID
            for node_id in sorted(graph.nodes.keys()):
                node = graph.nodes[node_id]
                f.write(f"{node.id}:({node.lat:.6f},{node.lng:.6f})\n")

            # Write all unique edges sorted by u then v
            edges_written = set()
            for u in sorted(graph.nodes.keys()):
                node = graph.nodes[u]
                for v, cost in sorted(node.neighbors, key=lambda x: x[0]):
                    edge_key = (u, v)
                    edges_written.add(edge_key)
                    f.write(f"{u},{v},{int(round(cost))}\n")
        print(f"Generated '{map_file_path}' successfully.")
        return True
    except Exception as e:
        print(f"Warning: Failed to write {map_file_path}: {e}")
        return False


def main():
    # 1. Parse command-line arguments
    if len(sys.argv) < 5:
        print(
            "Usage: python A2B.py <origin_node> <destination_node> <time> <model_type>"
        )
        print("Example: python A2B.py 2000 2825 1100 LSTM")
        sys.exit(1)

    origin_str = sys.argv[1]
    dest_str = sys.argv[2]
    time_str = sys.argv[3]
    model_name = sys.argv[4]

    # 2. Input Validation
    # Validate Node IDs
    try:
        origin_node = int(origin_str)
        dest_node = int(dest_str)
    except ValueError:
        print("Error: Origin and Destination node IDs must be integers.")
        sys.exit(1)

    # Validate time format (HHMM)
    if len(time_str) != 4 or not time_str.isdigit():
        print("Error: Time must be in 24-hour HHMM format (e.g. 1100).")
        sys.exit(1)

    hour = int(time_str[:2])
    minute = int(time_str[2:])
    if not (0 <= hour <= 23) or not (0 <= minute <= 59):
        print("Error: Invalid time. Hour must be 00-23 and Minute must be 00-59.")
        sys.exit(1)

    success = run_routing_and_prediction(origin_node, dest_node, time_str, model_name)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
