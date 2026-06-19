# src/routing/a_star.py
# This module implements the A* search pathfinding algorithm.

import heapq
import math


def heuristic(node_coords, dest_coords, max_speed=60.0):
    """
    Calculate the geographic distance (Haversine distance)
    between node_coords (lat1, lon1) and dest_coords (lat2, lon2).
    Divide this distance by the speed limit (60 km/h) to get a lower bound of travel time.
    This ensures the heuristic is admissible (never overestimates the true travel time).
    """
    lat1, lon1 = node_coords
    lat2, lon2 = dest_coords

    # Earth radius in km
    R = 6371.0

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_km = R * c

    # Time in hours = distance / speed
    # Time in seconds = (distance / speed) * 3600
    travel_time_seconds = (distance_km / max_speed) * 3600.0
    return travel_time_seconds


def a_star_search(graph, start_node_id, dest_node_id, ignored_nodes=None, ignored_edges=None):
    """
    Run A* search algorithm from start_node_id to dest_node_id with support for node/edge exclusion.

    :param graph: RoadNetworkGraph instance
    :param start_node_id: starting node ID
    :param dest_node_id: destination node ID
    :param ignored_nodes: set of node IDs to ignore
    :param ignored_edges: set of (u, v) tuples to ignore
    :return: path (list of node IDs), final travel time (float seconds)
    """
    if start_node_id not in graph.nodes or dest_node_id not in graph.nodes:
        return None, None

    if ignored_nodes and (start_node_id in ignored_nodes or dest_node_id in ignored_nodes):
        return None, None

    start_node = graph.nodes[start_node_id]
    dest_node = graph.nodes[dest_node_id]

    dest_coords = (dest_node.lat, dest_node.lng)

    # pq elements: (f_score, g_score, current_node_id, path)
    pq = []

    start_h = heuristic((start_node.lat, start_node.lng), dest_coords)
    heapq.heappush(pq, (start_h, 0.0, start_node_id, [start_node_id]))

    # Keep track of minimum g_score (travel time) to each visited node
    g_scores = {start_node_id: 0.0}

    while pq:
        f, g, curr_id, path = heapq.heappop(pq)

        if curr_id == dest_node_id:
            return path, g

        if g > g_scores.get(curr_id, float("inf")):
            continue

        curr_node = graph.nodes[curr_id]
        for neighbor_id, cost in curr_node.neighbors:
            if ignored_nodes and neighbor_id in ignored_nodes:
                continue
            if ignored_edges and ((curr_id, neighbor_id) in ignored_edges or (neighbor_id, curr_id) in ignored_edges):
                continue

            new_g = g + cost

            if new_g < g_scores.get(neighbor_id, float("inf")):
                g_scores[neighbor_id] = new_g
                neighbor_node = graph.nodes[neighbor_id]
                h = heuristic((neighbor_node.lat, neighbor_node.lng), dest_coords)
                f_new = new_g + h
                heapq.heappush(pq, (f_new, new_g, neighbor_id, path + [neighbor_id]))

    return None, None


def get_path_cost(graph, path):
    """
    Calculate the total cost of a path in the road network graph.
    """
    cost = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_cost = None
        for neighbor_id, c in graph.nodes[u].neighbors:
            if neighbor_id == v:
                edge_cost = c
                break
        if edge_cost is None:
            return float('inf')
        cost += edge_cost
    return cost


def yen_k_shortest_paths(graph, start_node_id, dest_node_id, k=3):
    """
    Find the K shortest loopless paths from start_node_id to dest_node_id using Yen's algorithm.

    :return: list of (path, cost) tuples
    """
    if start_node_id not in graph.nodes or dest_node_id not in graph.nodes:
        return []

    # Find the first shortest path
    path, cost = a_star_search(graph, start_node_id, dest_node_id)
    if not path:
        return []

    A = [(path, cost)]
    B = []  # Candidate paths: list of (path, cost) tuples

    for ki in range(1, k):
        prev_path = A[-1][0]
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i + 1]

            ignored_edges = set()
            ignored_nodes = set()

            for path_a, cost_a in A:
                if len(path_a) > i and path_a[:i + 1] == root_path:
                    ignored_edges.add((path_a[i], path_a[i + 1]))

            for node in root_path[:-1]:
                ignored_nodes.add(node)

            # Find the spur path
            spur_path, spur_cost = a_star_search(
                graph, spur_node, dest_node_id,
                ignored_nodes=ignored_nodes,
                ignored_edges=ignored_edges
            )

            if spur_path:
                total_path = root_path[:-1] + spur_path
                total_cost = get_path_cost(graph, total_path)
                
                candidate = (total_path, total_cost)
                if not any(np == total_path for np, nc in A) and not any(np == total_path for np, nc in B):
                    B.append(candidate)

        if not B:
            break

        # Sort candidate paths by cost
        B.sort(key=lambda x: x[1])
        # Move the lowest cost path to A
        A.append(B.pop(0))

    return A
