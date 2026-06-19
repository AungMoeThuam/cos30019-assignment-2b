# visualize.py
# This script loads map.txt and cos30019_2b.geojson, stitches a live CartoDB Voyager map,
# and displays the street network with curved routes and A* pathfinding in Pygame.
# Features: Responsive full-screen toggling, traffic flow animations, and green-yellow-red congestion coloring.
# Usage: python visualize.py

import pygame
import os
import sys
import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

import math
import json
import urllib.request
import io
import pandas as pd
from PIL import Image
from src.routing.a_star import a_star_search, heuristic

# Fallback pixel coordinates for offline mode (static map.png)
FALLBACK_PIXELS = {
    2000: (415, 421),
    2820: (129, 98),
    2825: (272, 55),
    2827: (339, 24),
    3002: (113, 215),
    3120: (250, 258),
    3127: (342, 272),
    3180: (367, 108),
    3662: (117, 180),
    3682: (426, 340),
    4032: (267, 142),
    4043: (229, 395),
    4051: (303, 95),
    4057: (359, 157),
    4263: (107, 259),
    4266: (192, 249),
    4270: (142, 300),
    4321: (215, 134)
}

# Color palette
COLOR_BG = (10, 10, 12)
COLOR_PANEL_BG = (22, 22, 26)
COLOR_CARD_BG = (32, 32, 40)
COLOR_TEXT_PRIMARY = (245, 245, 250)
COLOR_TEXT_SECONDARY = (160, 160, 175)
COLOR_ACCENT = (0, 229, 255)      # Cyan Glow
COLOR_SOURCE = (57, 255, 20)      # Neon Lime Green
COLOR_DEST = (255, 7, 58)         # Neon Red
COLOR_NODE = (120, 120, 140)
COLOR_HOVER = (255, 255, 255)

class GraphNode:
    def __init__(self, node_id, lat, lng):
        self.id = node_id
        self.lat = lat
        self.lng = lng
        self.neighbors = []

class MapGraph:
    def __init__(self):
        self.nodes = {}

def load_map_txt(file_path="map.txt"):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run A2B.py first to generate it.")
        sys.exit(1)
        
    graph = MapGraph()
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if len(lines) < 2:
        print("Error: Invalid map.txt format. Missing start/dest node lines.")
        sys.exit(1)
        
    start_node = int(lines[0])
    dest_node = int(lines[1])
    
    for line in lines[2:]:
        if ":" in line:
            parts = line.split(":")
            nid = int(parts[0])
            coords_str = parts[1].replace("(", "").replace(")", "")
            lat_str, lng_str = coords_str.split(",")
            graph.nodes[nid] = GraphNode(nid, float(lat_str), float(lng_str))
        elif "," in line:
            parts = line.split(",")
            u = int(parts[0])
            v = int(parts[1])
            cost = float(parts[2])
            if u in graph.nodes and v in graph.nodes:
                if not any(n[0] == v for n in graph.nodes[u].neighbors):
                    graph.nodes[u].neighbors.append((v, cost))
                if not any(n[0] == u for n in graph.nodes[v].neighbors):
                    graph.nodes[v].neighbors.append((u, cost))
                
    return graph, start_node, dest_node

def save_map_txt(graph, start_node, dest_node, file_path="map.txt"):
    try:
        with open(file_path, "w") as f:
            f.write(f"{start_node}\n")
            f.write(f"{dest_node}\n")
            for nid in sorted(graph.nodes.keys()):
                node = graph.nodes[nid]
                f.write(f"{node.id}:({node.lat:.6f},{node.lng:.6f})\n")
            
            edges_written = set()
            for u in sorted(graph.nodes.keys()):
                node = graph.nodes[u]
                for v, cost in sorted(node.neighbors, key=lambda x: x[0]):
                    edge_key = (u, v)
                    edges_written.add(edge_key)
                    f.write(f"{u},{v},{int(round(cost))}\n")
    except Exception as e:
        print(f"Warning: Failed to save changes back to map.txt: {e}")

def load_geojson_data(file_path="cos30019_2b.geojson"):
    visual_coords = {}
    curved_edges = {}
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Falling back to offline coords.")
        return visual_coords, curved_edges
        
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            
        features = data.get("features", [])
        
        # Pass 1: Parse all Point features to establish visual node coordinates
        for feature in features:
            geom = feature.get("geometry", {})
            props = feature.get("properties", {})
            if not geom or not props:
                continue
            gtype = geom.get("type")
            name = props.get("name")
            if gtype == "Point" and name:
                try:
                    nid = int(name)
                    lng, lat = geom["coordinates"]
                    visual_coords[nid] = (lat, lng)
                except ValueError:
                    pass
                    
        # Pass 2: Parse all LineString features and align coordinate directions with node endpoints
        for feature in features:
            geom = feature.get("geometry", {})
            props = feature.get("properties", {})
            if not geom or not props:
                continue
            gtype = geom.get("type")
            name = props.get("name")
            if gtype == "LineString" and name:
                parts = name.split("-")
                if len(parts) == 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        coords = [(lat, lng) for lng, lat in geom["coordinates"]]
                        
                        if u in visual_coords and v in visual_coords:
                            lat_u, lng_u = visual_coords[u]
                            # Measure distance from first/last coordinates of the line to u's coordinate
                            dist_start_u = (coords[0][0] - lat_u)**2 + (coords[0][1] - lng_u)**2
                            dist_end_u = (coords[-1][0] - lat_u)**2 + (coords[-1][1] - lng_u)**2
                            
                            if dist_end_u < dist_start_u:
                                # coords[-1] is closer to u, meaning coordinates are listed from v to u.
                                # Reverse them for (u, v), keep as-is for (v, u)
                                curved_edges[(u, v)] = coords[::-1]
                                curved_edges[(v, u)] = coords
                            else:
                                # coords[0] is closer to u, meaning coordinates are listed from u to v.
                                # Keep as-is for (u, v), reverse them for (v, u)
                                curved_edges[(u, v)] = coords
                                curved_edges[(v, u)] = coords[::-1]
                        else:
                            curved_edges[(u, v)] = coords
                            curved_edges[(v, u)] = coords[::-1]
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error reading GeoJSON: {e}")
        
    return visual_coords, curved_edges

def load_edge_distances(edges_csv="data/processed/edges.csv"):
    distances = {}
    if not os.path.exists(edges_csv):
        return distances
    try:
        df = pd.read_csv(edges_csv)
        df = df.dropna(subset=["from_site", "to_site"])
        for _, row in df.iterrows():
            u = int(row["from_site"])
            v = int(row["to_site"])
            dist = float(row["travel_distance_km"])
            distances[(u, v)] = dist
            distances[(v, u)] = dist
    except Exception as e:
        print(f"Error loading edges.csv: {e}")
    return distances

def get_congestion_color(cost, distance):
    """
    Interpolates segment color from Green -> Yellow -> Red based on speed/congestion.
    Free flow (60 km/h) = 1.0 (Green). Moderate delay = 1.6 (Yellow). Severe congestion = 2.2+ (Red).
    """
    base_time = (distance / 60.0) * 3600.0  # travel time in seconds at 60 km/h free-flow
    if base_time <= 0:
        return (57, 255, 20)  # Green fallback
        
    ratio = cost / base_time
    factor = min(1.0, max(0.0, (ratio - 1.0) / 1.2))
    
    # Core colors
    c_green = (57, 255, 20)
    c_yellow = (255, 215, 0)
    c_red = (255, 7, 58)
    
    if factor < 0.5:
        # Green to Yellow
        t = factor / 0.5
        r = int(c_green[0] + t * (c_yellow[0] - c_green[0]))
        g = int(c_green[1] + t * (c_yellow[1] - c_green[1]))
        b = int(c_green[2] + t * (c_yellow[2] - c_green[2]))
    else:
        # Yellow to Red
        t = (factor - 0.5) / 0.5
        r = int(c_yellow[0] + t * (c_red[0] - c_yellow[0]))
        g = int(c_yellow[1] + t * (c_red[1] - c_yellow[1]))
        b = int(c_yellow[2] + t * (c_red[2] - c_yellow[2]))
        
    return (r, g, b)

def get_route_gradient_color(index, total):
    """
    Interpolates active route color from Green -> Yellow -> Red based on path index.
    """
    if total <= 1:
        return (57, 255, 20)  # Green fallback
    t = index / (total - 1)
    
    c_green = (57, 255, 20)
    c_yellow = (255, 215, 0)
    c_red = (255, 7, 58)
    
    if t < 0.5:
        # Green to Yellow
        factor = t / 0.5
        r = int(c_green[0] + factor * (c_yellow[0] - c_green[0]))
        g = int(c_green[1] + factor * (c_yellow[1] - c_green[1]))
        b = int(c_green[2] + factor * (c_yellow[2] - c_green[2]))
    else:
        # Yellow to Red
        factor = (t - 0.5) / 0.5
        r = int(c_yellow[0] + factor * (c_red[0] - c_yellow[0]))
        g = int(c_yellow[1] + factor * (c_red[1] - c_yellow[1]))
        b = int(c_yellow[2] + factor * (c_red[2] - c_yellow[2]))
        
    return (r, g, b)

def latlng_to_tile_float(lat, lng, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = (lng + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return x, y

def get_live_osm_map(visual_coords, zoom=15):
    if not visual_coords:
        return None
        
    lats = [c[0] for c in visual_coords.values()]
    lngs = [c[1] for c in visual_coords.values()]
    
    min_lat, max_lat = min(lats) - 0.003, max(lats) + 0.003
    min_lng, max_lng = min(lngs) - 0.003, max(lngs) + 0.003
    
    x_min_f, y_min_f = latlng_to_tile_float(max_lat, min_lng, zoom)
    x_max_f, y_max_f = latlng_to_tile_float(min_lat, max_lng, zoom)
    
    min_xtile, max_xtile = int(x_min_f), int(x_max_f)
    min_ytile, max_ytile = int(y_min_f), int(y_max_f)
    
    num_cols = max_xtile - min_xtile + 1
    num_rows = max_ytile - min_ytile + 1
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    if os.path.exists("osm_map.png"):
        return min_xtile, min_ytile, num_cols, num_rows
        
    print("Fetching live CartoDB Voyager tiles...")
    stitched_img = Image.new("RGB", (num_cols * 256, num_rows * 256))
    
    success = True
    for r, ytile in enumerate(range(min_ytile, max_ytile + 1)):
        for c, xtile in enumerate(range(min_xtile, max_xtile + 1)):
            url = f"https://basemaps.cartocdn.com/rastertiles/voyager/{zoom}/{xtile}/{ytile}.png"
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    tile_data = response.read()
                    tile_image = Image.open(io.BytesIO(tile_data))
                    stitched_img.paste(tile_image, (c * 256, r * 256))
            except Exception as e:
                print(f"Error downloading tile {xtile},{ytile}: {e}")
                success = False
                break
        if not success:
            break
            
    if success:
        stitched_img.save("osm_map.png")
        print("Live CartoDB Voyager map generated!")
        return min_xtile, min_ytile, num_cols, num_rows
    else:
        return None

def main():
    # Parse arguments or prompt user if map.txt doesn't exist or if requested
    run_prediction = False
    origin_node = None
    dest_node = None
    time_str = None
    model_name = None

    if len(sys.argv) >= 5:
        # CLI arguments provided for prediction
        try:
            origin_node = int(sys.argv[1])
            dest_node = int(sys.argv[2])
            time_str = sys.argv[3]
            model_name = sys.argv[4].upper()
            run_prediction = True
        except ValueError:
            print("Error: Invalid CLI arguments. Origin and destination must be integers.")
            sys.exit(1)
    else:
        # No CLI arguments provided. Check map.txt
        map_exists = os.path.exists("map.txt")
        choice = 'n'
        if map_exists:
            print("Found existing 'map.txt'.")
            try:
                choice = input("Do you want to run a new prediction first? (y/n) [Default: n]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                choice = 'n'
                
        if not map_exists or choice == 'y':
            print("\n--- Route prediction configuration ---")
            
            # Origin input
            while True:
                try:
                    origin_input = input("Enter Origin Node (default: 2000): ").strip()
                    if not origin_input:
                        origin_node = 2000
                        break
                    origin_node = int(origin_input)
                    break
                except ValueError:
                    print("Invalid node ID. Please enter an integer.")
                    
            # Destination input
            while True:
                try:
                    dest_input = input("Enter Destination Node (default: 2825): ").strip()
                    if not dest_input:
                        dest_node = 2825
                        break
                    dest_node = int(dest_input)
                    break
                except ValueError:
                    print("Invalid node ID. Please enter an integer.")
                    
            # Time input
            while True:
                time_input = input("Enter Departure Time in HHMM format (default: 1100): ").strip()
                if not time_input:
                    time_str = "1100"
                    break
                if len(time_input) == 4 and time_input.isdigit():
                    hour = int(time_input[:2])
                    minute = int(time_input[2:])
                    if 0 <= hour <= 23 and 0 <= minute <= 59:
                        time_str = time_input
                        break
                print("Invalid time format. Please enter in 24-hour HHMM format (e.g. 1100).")
                
            # Model input
            while True:
                model_input = input("Enter Model (LSTM, GRU, RANDOM; default: LSTM): ").strip().upper()
                if not model_input:
                    model_name = "LSTM"
                    break
                if model_input in ["LSTM", "GRU", "RANDOM", "RF", "RANDOM_FOREST"]:
                    model_name = model_input
                    break
                print("Invalid model selection. Choose from LSTM, GRU, RANDOM, RF, RANDOM_FOREST.")
                
            run_prediction = True

    if run_prediction:
        from A2B import run_routing_and_prediction
        success = run_routing_and_prediction(origin_node, dest_node, time_str, model_name, "map.txt")
        if not success:
            print("Failed to run prediction. Exiting.")
            sys.exit(1)

    graph, start_node, dest_node = load_map_txt("map.txt")
    visual_coords, curved_edges = load_geojson_data("cos30019_2b.geojson")
    edge_distances = load_edge_distances("data/processed/edges.csv")
    
    pygame.init()
    pygame.font.init()
    
    zoom = 15
    osm_params = get_live_osm_map(visual_coords, zoom)
    
    window_w, window_h = 1100, 750
    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    pygame.display.set_caption("TBRGS Map Visualizer (OpenStreetMap)")
    clock = pygame.time.Clock()
    
    is_fullscreen = False
    raw_map_image = None
    map_file_name = "osm_map.png" if osm_params else "map.png"
    if os.path.exists(map_file_name):
        try:
            raw_map_image = pygame.image.load(map_file_name).convert()
        except Exception as e:
            print(f"Error loading map background: {e}")
            
    if 'time_str' not in locals() or time_str is None:
        time_str = "1100"
    if 'model_name' not in locals() or model_name is None:
        model_name = "LSTM"

    current_start = start_node
    current_dest = dest_node
    current_time = time_str
    current_model = model_name
    selected_route_index = 0
    
    input_start_str = str(current_start)
    input_dest_str = str(current_dest)
    input_time_str = current_time
    
    status_message = ""
    status_color = COLOR_SOURCE
    active_field = None
    
    current_paths = []
    
    particles = []
    particle_timer = 0
    
    def update_route():
        nonlocal current_paths, particles, graph, selected_route_index
        selected_route_index = 0
        from A2B import run_routing_and_prediction
        # Run prediction dynamically with new parameters and save to map.txt
        success = run_routing_and_prediction(current_start, current_dest, current_time, current_model, "map.txt")
        if success:
            # Reload graph weights from the updated map.txt
            graph, _, _ = load_map_txt("map.txt")
            
        from src.routing.a_star import yen_k_shortest_paths
        current_paths = yen_k_shortest_paths(graph, current_start, current_dest, k=3)
        particles = []
        
    def trigger_update():
        nonlocal current_start, current_dest, current_time, current_model, status_message, status_color
        
        # 1. Validate Source Node
        try:
            val_start = int(input_start_str)
            if val_start not in graph.nodes:
                status_message = "Error: Source ID not found"
                status_color = COLOR_DEST
                return
        except ValueError:
            status_message = "Error: Source ID must be integer"
            status_color = COLOR_DEST
            return
            
        # 2. Validate Dest Node
        try:
            val_dest = int(input_dest_str)
            if val_dest not in graph.nodes:
                status_message = "Error: Dest ID not found"
                status_color = COLOR_DEST
                return
        except ValueError:
            status_message = "Error: Dest ID must be integer"
            status_color = COLOR_DEST
            return
            
        # 3. Validate Time format
        if len(input_time_str) != 4 or not input_time_str.isdigit():
            status_message = "Error: Time must be HHMM"
            status_color = COLOR_DEST
            return
            
        val_hour = int(input_time_str[:2])
        val_min = int(input_time_str[2:])
        if not (0 <= val_hour <= 23) or not (0 <= val_min <= 59):
            status_message = "Error: Time ranges invalid"
            status_color = COLOR_DEST
            return
            
        # 4. Commit changes and run update
        current_start = val_start
        current_dest = val_dest
        current_time = input_time_str
        
        status_message = "Updating routes..."
        status_color = (255, 215, 0)
        
        # Re-run route prediction and paths
        update_route()
        
        status_message = "Routes updated successfully!"
        status_color = COLOR_SOURCE
        
    update_route()
    
    running = True
    while running:
        w_w, w_h = screen.get_size()
        panel_w = int(w_w * 0.3)
        if panel_w < 300:
            panel_w = 300
        elif panel_w > 400:
            panel_w = 400
            
        map_w = w_w - panel_w
        map_h = w_h
        
        font_size_body = max(13, int(w_h * 0.02))
        font_size_title = max(20, int(w_h * 0.032))
        font_size_header = max(16, int(w_h * 0.024))
        
        try:
            font_title = pygame.font.SysFont("Helvetica", font_size_title, bold=True)
            font_header = pygame.font.SysFont("Helvetica", font_size_header, bold=True)
            font_body = pygame.font.SysFont("Helvetica", font_size_body)
            font_small = pygame.font.SysFont("Helvetica", max(11, int(w_h * 0.016)))
        except:
            font_title = pygame.font.Font(None, font_size_title + 4)
            font_header = pygame.font.Font(None, font_size_header + 4)
            font_body = pygame.font.Font(None, font_size_body + 2)
            font_small = pygame.font.Font(None, max(11, int(w_h * 0.016)))
            
        if osm_params:
            min_xtile, min_ytile, num_cols, num_rows = osm_params
            orig_map_w = num_cols * 256
            orig_map_h = num_rows * 256
            scale_x = map_w / orig_map_w
            scale_y = map_h / orig_map_h
            
            node_pixels = {}
            for nid, (lat, lng) in visual_coords.items():
                xf, yf = latlng_to_tile_float(lat, lng, zoom)
                px = (xf - min_xtile) * 256
                py = (yf - min_ytile) * 256
                node_pixels[nid] = (int(px * scale_x), int(py * scale_y))
        else:
            scale_x = map_w / 479.0
            scale_y = map_h / 447.0
            node_pixels = {nid: (int(x * scale_x), int(y * scale_y)) for nid, (x, y) in FALLBACK_PIXELS.items()}
            
        particle_timer += 1
        active_path = current_paths[selected_route_index][0] if (current_paths and len(current_paths) > selected_route_index) else None
        if active_path and len(active_path) >= 2:
            if particle_timer % 15 == 0:
                particles.append({
                    "segment": 0,
                    "progress": 0.0,
                    "speed": 0.03
                })
                
            active_particles = []
            for p in particles:
                p["progress"] += p["speed"]
                if p["progress"] >= 1.0:
                    p["progress"] = 0.0
                    p["segment"] += 1
                if p["segment"] < len(active_path) - 1:
                    active_particles.append(p)
            particles = active_particles
        else:
            particles = []
            
        # Calculate sidebar control rect positions
        card_y = int(w_h * 0.03) + int(w_h * 0.07) # starts right after title height
        y_start_node = card_y + 36
        y_dest_node = card_y + 70
        y_time = card_y + 104
        y_model = card_y + 138
        y_status = card_y + 172
        
        if status_message:
            y_update = card_y + 192
            card_h = 234
        else:
            y_update = card_y + 172
            card_h = 214
            
        rect_input_start = pygame.Rect(map_w + 130, y_start_node, panel_w - 160, 26)
        rect_input_dest = pygame.Rect(map_w + 130, y_dest_node, panel_w - 160, 26)
        rect_input_time = pygame.Rect(map_w + 130, y_time, panel_w - 160, 26)
        
        rect_btn_lstm = pygame.Rect(map_w + 130, y_model, 50, 24)
        rect_btn_gru = pygame.Rect(map_w + 185, y_model, 50, 24)
        rect_btn_rf = pygame.Rect(map_w + 240, y_model, 50, 24)
        
        rect_btn_update = pygame.Rect(map_w + 25, y_update, panel_w - 50, 30)

        # Calculate Route Analytics rows clickable bounding boxes
        stats_card_y = card_y + card_h + 15 + int(w_h * 0.04)
        y_pos = stats_card_y + 10
        row_h = int(w_h * 0.065)
        rect_route_rows = []
        for p_idx in range(min(3, len(current_paths))):
            rect_route_rows.append(pygame.Rect(map_w + 20, y_pos, panel_w - 40, row_h))
            y_pos += row_h + 8

        mouse_pos = pygame.mouse.get_pos()
        hovered_node = None
        hovered_edge = None
        
        for node_id, (px, py) in node_pixels.items():
            dist = math.hypot(mouse_pos[0] - px, mouse_pos[1] - py)
            if dist <= 14:
                hovered_node = node_id
                break
                
        if hovered_node is None and mouse_pos[0] < map_w and mouse_pos[1] < map_h:
            min_dist_to_edge = 10.0
            for node_id, node in graph.nodes.items():
                p1 = node_pixels.get(node_id)
                if not p1: continue
                for neighbor_id, cost in node.neighbors:
                    p2 = node_pixels.get(neighbor_id)
                    if not p2: continue
                    
                    coords = curved_edges.get((node_id, neighbor_id))
                    if coords and osm_params:
                        points = []
                        for lat, lng in coords:
                            xf, yf = latlng_to_tile_float(lat, lng, zoom)
                            px = (xf - min_xtile) * 256
                            py = (yf - min_ytile) * 256
                            points.append((int(px * scale_x), int(py * scale_y)))
                            
                        for i in range(len(points) - 1):
                            x1, y1 = points[i]
                            x2, y2 = points[i+1]
                            x0, y0 = mouse_pos
                            len_sq = (x2 - x1)**2 + (y2 - y1)**2
                            if len_sq == 0: continue
                            t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / len_sq))
                            proj_x = x1 + t * (x2 - x1)
                            proj_y = y1 + t * (y2 - y1)
                            d = math.hypot(x0 - proj_x, y0 - proj_y)
                            if d < min_dist_to_edge:
                                min_dist_to_edge = d
                                hovered_edge = (node_id, neighbor_id, cost)
                    else:
                        x0, y0 = mouse_pos
                        x1, y1 = p1
                        x2, y2 = p2
                        len_sq = (x2 - x1)**2 + (y2 - y1)**2
                        if len_sq == 0: continue
                        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / len_sq))
                        proj_x = x1 + t * (x2 - x1)
                        proj_y = y1 + t * (y2 - y1)
                        d = math.hypot(x0 - proj_x, y0 - proj_y)
                        if d < min_dist_to_edge:
                            min_dist_to_edge = d
                            hovered_edge = (node_id, neighbor_id, cost)
                            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    mx, my = event.pos
                    if mx >= map_w:
                        # Sidebar controls clicks
                        if rect_input_start.collidepoint(mx, my):
                            active_field = "start"
                        elif rect_input_dest.collidepoint(mx, my):
                            active_field = "dest"
                        elif rect_input_time.collidepoint(mx, my):
                            active_field = "time"
                        elif rect_btn_lstm.collidepoint(mx, my):
                            current_model = "LSTM"
                            active_field = None
                        elif rect_btn_gru.collidepoint(mx, my):
                            current_model = "GRU"
                            active_field = None
                        elif rect_btn_rf.collidepoint(mx, my):
                            current_model = "RF"
                            active_field = None
                        elif rect_btn_update.collidepoint(mx, my):
                            active_field = None
                            trigger_update()
                        else:
                            # Check if one of the route rows was clicked
                            for p_idx, r_rect in enumerate(rect_route_rows):
                                if r_rect.collidepoint(mx, my):
                                    selected_route_index = p_idx
                                    particles = [] # Reset particles on change
                                    break
                            active_field = None
                    else:
                        active_field = None
                else:
                    active_field = None
            elif event.type == pygame.KEYDOWN:
                if active_field is not None:
                    if event.key == pygame.K_BACKSPACE:
                        if active_field == "start":
                            input_start_str = input_start_str[:-1]
                        elif active_field == "dest":
                            input_dest_str = input_dest_str[:-1]
                        elif active_field == "time":
                            input_time_str = input_time_str[:-1]
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        active_field = None
                        trigger_update()
                    elif event.key == pygame.K_ESCAPE:
                        active_field = None
                    else:
                        char = event.unicode
                        if char and char.isprintable():
                            if active_field == "start":
                                if len(input_start_str) < 8 and char.isdigit():
                                    input_start_str += char
                            elif active_field == "dest":
                                if len(input_dest_str) < 8 and char.isdigit():
                                    input_dest_str += char
                            elif active_field == "time":
                                if len(input_time_str) < 4 and char.isdigit():
                                    input_time_str += char
                else:
                    if event.key == pygame.K_f or event.key == pygame.K_F11:
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
                        
        screen.fill(COLOR_BG)
        
        if raw_map_image:
            scaled_map = pygame.transform.smoothscale(raw_map_image, (map_w, map_h))
            screen.blit(scaled_map, (0, 0))
        else:
            pygame.draw.rect(screen, (30, 30, 35), (0, 0, map_w, map_h))
            
        # Draw Network Edges with Congestion Heatmap Colors
        drawn_edges = set()
        for node_id, node in graph.nodes.items():
            p1 = node_pixels.get(node_id)
            if not p1: continue
            for neighbor_id, cost in node.neighbors:
                edge_key = tuple(sorted((node_id, neighbor_id)))
                if edge_key in drawn_edges: continue
                drawn_edges.add(edge_key)
                
                p2 = node_pixels.get(neighbor_id)
                if not p2: continue
                
                # Check path inclusion
                is_in_active_path = False
                is_in_faded_path = False
                
                if current_paths:
                    active_path = current_paths[selected_route_index][0] if len(current_paths) > selected_route_index else None
                    if active_path:
                        for idx in range(len(active_path) - 1):
                            if (active_path[idx] == node_id and active_path[idx+1] == neighbor_id) or \
                               (active_path[idx] == neighbor_id and active_path[idx+1] == node_id):
                                is_in_active_path = True
                                break
                    
                    if not is_in_active_path:
                        for p_idx, path_data in enumerate(current_paths):
                            if p_idx == selected_route_index:
                                continue
                            fpath = path_data[0]
                            for idx in range(len(fpath) - 1):
                                if (fpath[idx] == node_id and fpath[idx+1] == neighbor_id) or \
                                   (fpath[idx] == neighbor_id and fpath[idx+1] == node_id):
                                    is_in_faded_path = True
                                    break
                            if is_in_faded_path:
                                break
                            
                # Get the edge distance
                dist = edge_distances.get((node_id, neighbor_id))
                if not dist:
                    # Fallback to geodesic Haversine distance
                    lat1, lon1 = graph.nodes[node_id].lat, graph.nodes[node_id].lng
                    lat2, lon2 = graph.nodes[neighbor_id].lat, graph.nodes[neighbor_id].lng
                    dist = heuristic((lat1, lon1), (lat2, lon2)) / 3600.0 * 60.0
                    
                # Congestion color
                congestion_color = get_congestion_color(cost, dist)
                
                # Render style: active path is thick, faded path is medium, background is thin
                if is_in_active_path:
                    thickness = 7
                    color = congestion_color
                elif is_in_faded_path:
                    thickness = 5
                    color = (
                        int(congestion_color[0] * 0.75 + COLOR_BG[0] * 0.25),
                        int(congestion_color[1] * 0.75 + COLOR_BG[1] * 0.25),
                        int(congestion_color[2] * 0.75 + COLOR_BG[2] * 0.25)
                    )
                else:
                    thickness = 3
                    color = (
                        int(congestion_color[0] * 0.5 + COLOR_BG[0] * 0.5),
                        int(congestion_color[1] * 0.5 + COLOR_BG[1] * 0.5),
                        int(congestion_color[2] * 0.5 + COLOR_BG[2] * 0.5)
                    )
                
                coords = curved_edges.get((node_id, neighbor_id))
                if coords and osm_params:
                    points = []
                    for lat, lng in coords:
                        xf, yf = latlng_to_tile_float(lat, lng, zoom)
                        px = (xf - min_xtile) * 256
                        py = (yf - min_ytile) * 256
                        points.append((int(px * scale_x), int(py * scale_y)))
                    if len(points) >= 2:
                        if is_in_active_path:
                            # Draw soft glow matching congestion color behind active path
                            pygame.draw.lines(screen, (color[0], color[1], color[2]), False, points, thickness + 4)
                        pygame.draw.lines(screen, color, False, points, thickness)
                else:
                    if is_in_active_path:
                        pygame.draw.line(screen, color, p1, p2, thickness + 4)
                    pygame.draw.line(screen, color, p1, p2, thickness)
                    
        # Draw Moving Traffic Particles
        active_path = current_paths[0][0] if (current_paths and len(current_paths) > 0) else None
        for p in particles:
            seg_idx = p["segment"]
            progress = p["progress"]
            
            # Defensive bounds checking to prevent IndexError
            if not active_path or seg_idx >= len(active_path) - 1:
                continue
                
            u_id = active_path[seg_idx]
            v_id = active_path[seg_idx+1]
            
            coords = curved_edges.get((u_id, v_id))
            if coords and osm_params:
                points = []
                for lat, lng in coords:
                    xf, yf = latlng_to_tile_float(lat, lng, zoom)
                    px = (xf - min_xtile) * 256
                    py = (yf - min_ytile) * 256
                    points.append((int(px * scale_x), int(py * scale_y)))
                
                num_pts = len(points)
                if num_pts >= 2:
                    total_progress = progress * (num_pts - 1)
                    idx = int(total_progress)
                    t_val = total_progress - idx
                    x1, y1 = points[idx]
                    x2, y2 = points[idx+1]
                    part_x = x1 + t_val * (x2 - x1)
                    part_y = y1 + t_val * (y2 - y1)
                else:
                    continue
            else:
                p1 = node_pixels[u_id]
                p2 = node_pixels[v_id]
                part_x = p1[0] + progress * (p2[0] - p1[0])
                part_y = p1[1] + progress * (p2[1] - p1[1])
                
            pygame.draw.circle(screen, (255, 255, 255), (int(part_x), int(part_y)), 4)
            pygame.draw.circle(screen, COLOR_ACCENT, (int(part_x), int(part_y)), 7, 1)

        # Draw Intersections/Nodes
        for node_id, (px, py) in node_pixels.items():
            if node_id not in graph.nodes: continue
            
            if node_id == current_start:
                color = COLOR_SOURCE
                radius = 11
                pygame.draw.circle(screen, (57, 255, 20, 60), (px, py), radius + 8, 2)
            elif node_id == current_dest:
                color = COLOR_DEST
                radius = 11
                pygame.draw.circle(screen, (255, 7, 58, 60), (px, py), radius + 8, 2)
            elif current_paths and len(current_paths) > selected_route_index and node_id in current_paths[selected_route_index][0]:
                color = COLOR_ACCENT
                radius = 8
            else:
                color = COLOR_NODE
                radius = 6
                
            if node_id == hovered_node:
                pygame.draw.circle(screen, COLOR_HOVER, (px, py), radius + 3, 2)
                
            pygame.draw.circle(screen, color, (px, py), radius)
            
            # Render node ID badge for clear visibility on light map backgrounds
            node_id_str = str(node_id)
            lbl = font_small.render(node_id_str, True, COLOR_TEXT_PRIMARY)
            text_w, text_h = lbl.get_size()
            
            badge_rect = pygame.Rect(px + 12, py - text_h // 2, text_w + 8, text_h + 4)
            pygame.draw.rect(screen, (22, 22, 26), badge_rect, border_radius=4)
            
            outline_color = COLOR_ACCENT if node_id in [current_start, current_dest] else (100, 100, 110)
            pygame.draw.rect(screen, outline_color, badge_rect, 1, border_radius=4)
            
            screen.blit(lbl, (px + 16, py - text_h // 2 + 2))
            
        # Render Glassmorphism Side Panel
        panel_rect = pygame.Rect(map_w, 0, panel_w, w_h)
        pygame.draw.rect(screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(screen, (60, 60, 75), (map_w, 0), (map_w, w_h), 2)
        
        y_offset = int(w_h * 0.03)
        
        title_surf = font_title.render("TBRGS NAVIGATOR", True, COLOR_ACCENT)
        screen.blit(title_surf, (map_w + 20, y_offset))
        y_offset += int(w_h * 0.07)
        
        # Draw Route Settings Card background
        card_rect = pygame.Rect(map_w + 15, y_offset, panel_w - 30, card_h)
        pygame.draw.rect(screen, COLOR_CARD_BG, card_rect, border_radius=8)
        pygame.draw.rect(screen, (60, 60, 75), card_rect, 1, border_radius=8)
        
        # Header text
        lbl_header = font_header.render("Route Settings", True, COLOR_ACCENT)
        screen.blit(lbl_header, (map_w + 25, card_y + 10))
        
        # Row 1: Source ID
        lbl_src_title = font_body.render("Source ID:", True, COLOR_TEXT_PRIMARY)
        screen.blit(lbl_src_title, (map_w + 25, y_start_node + 3))
        
        box_src_color = COLOR_ACCENT if active_field == "start" else (80, 80, 95)
        pygame.draw.rect(screen, (20, 20, 24), rect_input_start, border_radius=4)
        pygame.draw.rect(screen, box_src_color, rect_input_start, 1, border_radius=4)
        
        # Draw text inside box
        txt_src_surf = font_body.render(input_start_str, True, COLOR_TEXT_PRIMARY)
        screen.blit(txt_src_surf, (rect_input_start.x + 8, rect_input_start.y + 3))
        
        # Draw cursor if active
        if active_field == "start" and (pygame.time.get_ticks() // 400) % 2 == 0:
            cursor_x = rect_input_start.x + 8 + font_body.size(input_start_str)[0]
            pygame.draw.line(screen, COLOR_TEXT_PRIMARY, (cursor_x, rect_input_start.y + 4), (cursor_x, rect_input_start.y + 22), 2)
            
        # Row 2: Dest ID
        lbl_dst_title = font_body.render("Dest ID:", True, COLOR_TEXT_PRIMARY)
        screen.blit(lbl_dst_title, (map_w + 25, y_dest_node + 3))
        
        box_dst_color = COLOR_ACCENT if active_field == "dest" else (80, 80, 95)
        pygame.draw.rect(screen, (20, 20, 24), rect_input_dest, border_radius=4)
        pygame.draw.rect(screen, box_dst_color, rect_input_dest, 1, border_radius=4)
        
        txt_dst_surf = font_body.render(input_dest_str, True, COLOR_TEXT_PRIMARY)
        screen.blit(txt_dst_surf, (rect_input_dest.x + 8, rect_input_dest.y + 3))
        
        if active_field == "dest" and (pygame.time.get_ticks() // 400) % 2 == 0:
            cursor_x = rect_input_dest.x + 8 + font_body.size(input_dest_str)[0]
            pygame.draw.line(screen, COLOR_TEXT_PRIMARY, (cursor_x, rect_input_dest.y + 4), (cursor_x, rect_input_dest.y + 22), 2)
            
        # Row 3: Time
        lbl_time_title = font_body.render("Time:", True, COLOR_TEXT_PRIMARY)
        screen.blit(lbl_time_title, (map_w + 25, y_time + 3))
        
        box_time_color = COLOR_ACCENT if active_field == "time" else (80, 80, 95)
        pygame.draw.rect(screen, (20, 20, 24), rect_input_time, border_radius=4)
        pygame.draw.rect(screen, box_time_color, rect_input_time, 1, border_radius=4)
        
        txt_time_surf = font_body.render(input_time_str, True, COLOR_TEXT_PRIMARY)
        screen.blit(txt_time_surf, (rect_input_time.x + 8, rect_input_time.y + 3))
        
        if active_field == "time" and (pygame.time.get_ticks() // 400) % 2 == 0:
            cursor_x = rect_input_time.x + 8 + font_body.size(input_time_str)[0]
            pygame.draw.line(screen, COLOR_TEXT_PRIMARY, (cursor_x, rect_input_time.y + 4), (cursor_x, rect_input_time.y + 22), 2)
            
        # Row 4: Model Selector Buttons
        lbl_model_title = font_body.render("Model:", True, COLOR_TEXT_PRIMARY)
        screen.blit(lbl_model_title, (map_w + 25, y_model + 3))
        
        for m_name, btn_rect in [("LSTM", rect_btn_lstm), ("GRU", rect_btn_gru), ("RF", rect_btn_rf)]:
            # Highlight selected
            is_sel = (current_model == m_name or (m_name == "RF" and current_model in ["RANDOM", "RANDOM_FOREST"]))
            btn_bg = COLOR_ACCENT if is_sel else (45, 45, 55)
            btn_fg = (10, 10, 15) if is_sel else COLOR_TEXT_SECONDARY
            
            pygame.draw.rect(screen, btn_bg, btn_rect, border_radius=4)
            pygame.draw.rect(screen, (80, 80, 95), btn_rect, 1, border_radius=4)
            
            lbl_btn = font_small.render(m_name, True, btn_fg)
            screen.blit(lbl_btn, (btn_rect.x + (btn_rect.width - lbl_btn.get_width()) // 2, btn_rect.y + 4))
            
        # Row 5: Status Message (if any)
        if status_message:
            lbl_status = font_small.render(status_message, True, status_color)
            screen.blit(lbl_status, (map_w + 25, y_status))
            
        # Row 6: UPDATE ROUTE Button
        mouse_x, mouse_y = pygame.mouse.get_pos()
        is_hovered = rect_btn_update.collidepoint(mouse_x, mouse_y)
        up_btn_bg = (0, 204, 242) if is_hovered else (0, 180, 216)
        
        pygame.draw.rect(screen, up_btn_bg, rect_btn_update, border_radius=6)
        lbl_update = font_body.render("UPDATE ROUTE", True, (255, 255, 255))
        screen.blit(lbl_update, (rect_btn_update.x + (rect_btn_update.width - lbl_update.get_width()) // 2, rect_btn_update.y + 5))
        
        y_offset += card_h + 15
        
        # Route Statistics Card
        stats_header = font_header.render("Route Analytics", True, COLOR_ACCENT)
        screen.blit(stats_header, (map_w + 20, y_offset))
        y_offset += int(w_h * 0.04)
        
        card_stats = pygame.Rect(map_w + 15, y_offset, panel_w - 30, int(w_h * 0.28))
        pygame.draw.rect(screen, COLOR_CARD_BG, card_stats, border_radius=8)
        pygame.draw.rect(screen, (60, 60, 75), card_stats, 1, border_radius=8)
        
        if current_paths:
            # Draw stats for all paths in current_paths as interactive rows
            for p_idx, path_data in enumerate(current_paths[:3]):
                fpath, travel_time = path_data
                mins = int(travel_time // 60)
                secs = int(travel_time % 60)
                
                # Check selection state
                is_sel = (p_idx == selected_route_index)
                
                # Draw selection box
                row_rect = rect_route_rows[p_idx]
                if is_sel:
                    pygame.draw.rect(screen, (50, 50, 65), row_rect, border_radius=6)
                    pygame.draw.rect(screen, COLOR_ACCENT, row_rect, 1, border_radius=6)
                else:
                    pygame.draw.rect(screen, (40, 40, 48), row_rect, border_radius=6)
                    pygame.draw.rect(screen, (70, 70, 85), row_rect, 1, border_radius=6)
                    
                label_prefix = "Best Route" if p_idx == 0 else f"{p_idx+1}nd Route" if p_idx == 1 else f"{p_idx+1}rd Route"
                text_color = COLOR_ACCENT if is_sel else COLOR_SOURCE if p_idx == 0 else COLOR_TEXT_PRIMARY
                
                # Draw radio checkmark circle
                radio_x = row_rect.x + 15
                radio_y = row_rect.y + row_rect.height // 2
                pygame.draw.circle(screen, (100, 100, 120), (radio_x, radio_y), 6, 1)
                if is_sel:
                    pygame.draw.circle(screen, COLOR_ACCENT, (radio_x, radio_y), 3)
                    
                time_result = font_small.render(f"{label_prefix}: {mins} min {secs} s ({len(fpath)} nodes)", True, text_color)
                screen.blit(time_result, (row_rect.x + 30, row_rect.y + 6))
                
                # Render the path nodes briefly on a new line
                path_str = " -> ".join(map(str, fpath))
                words = path_str.split(" -> ")
                curr_line = ""
                lines = []
                for w in words:
                    test_line = curr_line + (" -> " if curr_line else "") + w
                    if font_small.size(test_line)[0] < row_rect.width - 45:
                        curr_line = test_line
                    else:
                        lines.append(curr_line)
                        curr_line = w
                if curr_line:
                    lines.append(curr_line)
                    
                if lines:
                    line_surf = font_small.render(lines[0], True, COLOR_TEXT_SECONDARY)
                    screen.blit(line_surf, (row_rect.x + 30, row_rect.y + 24))
                
            # Draw Route Gradient Progress Bar Legend at the bottom of the card
            bar_y = y_offset + int(w_h * 0.245)
            bar_x = map_w + 30
            bar_w = panel_w - 60
            bar_h = 6
            
            # Draw gradient bar
            for dx in range(bar_w):
                t = dx / bar_w
                if t < 0.5:
                    factor = t / 0.5
                    r = int(57 + factor * (255 - 57))
                    g = int(255 + factor * (215 - 255))
                    b = int(20 + factor * (0 - 20))
                else:
                    factor = (t - 0.5) / 0.5
                    r = int(255 + factor * (255 - 255))
                    g = int(215 + factor * (7 - 215))
                    b = int(0 + factor * (58 - 0))
                pygame.draw.line(screen, (r, g, b), (bar_x + dx, bar_y), (bar_x + dx, bar_y + bar_h))
                
            # Draw labels under the bar
            lbl_start = font_small.render("Low Volume", True, COLOR_SOURCE)
            lbl_mid = font_small.render("Moderate", True, (255, 215, 0))
            lbl_end = font_small.render("High Volume", True, COLOR_DEST)
            screen.blit(lbl_start, (bar_x, bar_y + bar_h + 2))
            screen.blit(lbl_mid, (bar_x + bar_w // 2 - lbl_mid.get_width() // 2, bar_y + bar_h + 2))
            screen.blit(lbl_end, (bar_x + bar_w - lbl_end.get_width(), bar_y + bar_h + 2))
        else:
            no_path = font_body.render("No path exists!", True, COLOR_DEST)
            screen.blit(no_path, (map_w + 30, y_offset + 15))
            
        y_offset += int(w_h * 0.31)
        
        # Interactive Controls Reference
        help_header = font_header.render("Controls Menu", True, COLOR_TEXT_SECONDARY)
        screen.blit(help_header, (map_w + 20, y_offset))
        y_offset += int(w_h * 0.04)
        
        helps = [
            "Click UPDATE ROUTE to calculate new paths.",
            "F / F11 key      : Toggle Full Screen Mode",
            "Node details save directly to map.txt"
        ]
        for h_text in helps:
            help_surf = font_small.render(h_text, True, COLOR_TEXT_SECONDARY)
            screen.blit(help_surf, (map_w + 20, y_offset))
            y_offset += int(w_h * 0.03)
            
        # Render Overlay Tooltips
        if hovered_node is not None and hovered_node in visual_coords:
            v_lat, v_lng = visual_coords[hovered_node]
            tooltip_rect = pygame.Rect(mouse_pos[0] + 15, mouse_pos[1] - 40, 200, 56)
            pygame.draw.rect(screen, (15, 15, 20), tooltip_rect, border_radius=4)
            pygame.draw.rect(screen, COLOR_ACCENT, tooltip_rect, 1, border_radius=4)
            t1 = font_small.render(f"Intersection ID: {hovered_node}", True, COLOR_TEXT_PRIMARY)
            t2 = font_small.render(f"Lat: {v_lat:.6f}", True, COLOR_TEXT_SECONDARY)
            t3 = font_small.render(f"Lng: {v_lng:.6f}", True, COLOR_TEXT_SECONDARY)
            screen.blit(t1, (mouse_pos[0] + 25, mouse_pos[1] - 35))
            screen.blit(t2, (mouse_pos[0] + 25, mouse_pos[1] - 21))
            screen.blit(t3, (mouse_pos[0] + 25, mouse_pos[1] - 9))
            
        elif hovered_edge is not None:
            u, v, cost = hovered_edge
            tooltip_rect = pygame.Rect(mouse_pos[0] + 15, mouse_pos[1] - 30, 180, 48)
            pygame.draw.rect(screen, (15, 15, 20), tooltip_rect, border_radius=4)
            pygame.draw.rect(screen, COLOR_ACCENT, tooltip_rect, 1, border_radius=4)
            
            # Find the distance
            dist = edge_distances.get((u, v))
            if not dist:
                lat1, lon1 = graph.nodes[u].lat, graph.nodes[u].lng
                lat2, lon2 = graph.nodes[v].lat, graph.nodes[v].lng
                dist = heuristic((lat1, lon1), (lat2, lon2)) / 3600.0 * 60.0
                
            # Speed = distance / time
            # speed_kmh = (dist / (cost / 3600))
            speed = (dist / (cost / 3600.0)) if cost > 0 else 0
            
            t1 = font_small.render(f"Segment: {u} -> {v}", True, COLOR_TEXT_PRIMARY)
            t2 = font_small.render(f"Travel Time: {int(round(cost))}s ({speed:.1f} km/h)", True, COLOR_TEXT_SECONDARY)
            screen.blit(t1, (mouse_pos[0] + 25, mouse_pos[1] - 25))
            screen.blit(t2, (mouse_pos[0] + 25, mouse_pos[1] - 12))

        pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()

if __name__ == "__main__":
    main()
