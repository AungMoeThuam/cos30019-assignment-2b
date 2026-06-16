# visualize.py
# This script loads map.txt and cos30019_2b.geojson, stitches a live CartoDB Voyager map,
# and displays the street network with curved routes and A* pathfinding in Pygame.
# Features: Responsive full-screen toggling, traffic flow animations, and green-yellow-red congestion coloring.
# Usage: python visualize.py

import pygame
import os
import sys
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
            
        for feature in data.get("features", []):
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
            elif gtype == "LineString" and name:
                parts = name.split("-")
                if len(parts) == 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        coords = [(lat, lng) for lng, lat in geom["coordinates"]]
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
            
    current_start = start_node
    current_dest = dest_node
    current_path = None
    current_travel_time = 0.0
    
    particles = []
    particle_timer = 0
    
    def update_route():
        nonlocal current_path, current_travel_time
        current_path, current_travel_time = a_star_search(graph, current_start, current_dest)
        save_map_txt(graph, current_start, current_dest)
        
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
        if current_path and len(current_path) >= 2:
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
                if p["segment"] < len(current_path) - 1:
                    active_particles.append(p)
            particles = active_particles
        else:
            particles = []
            
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
                if hovered_node is not None:
                    if event.button == 1: # Left click
                        current_start = hovered_node
                        update_route()
                    elif event.button == 3: # Right click
                        current_dest = hovered_node
                        update_route()
            elif event.type == pygame.KEYDOWN:
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
                is_in_path = False
                if current_path:
                    for idx in range(len(current_path) - 1):
                        if (current_path[idx] == node_id and current_path[idx+1] == neighbor_id) or \
                           (current_path[idx] == neighbor_id and current_path[idx+1] == node_id):
                            is_in_path = True
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
                
                # Render style: path is thick, background is thin
                thickness = 7 if is_in_path else 3
                
                # Reduce opacity of background edges slightly by blending with BG color
                if not is_in_path:
                    color = (
                        int(congestion_color[0] * 0.5 + COLOR_BG[0] * 0.5),
                        int(congestion_color[1] * 0.5 + COLOR_BG[1] * 0.5),
                        int(congestion_color[2] * 0.5 + COLOR_BG[2] * 0.5)
                    )
                else:
                    color = congestion_color
                
                coords = curved_edges.get((node_id, neighbor_id))
                if coords and osm_params:
                    points = []
                    for lat, lng in coords:
                        xf, yf = latlng_to_tile_float(lat, lng, zoom)
                        px = (xf - min_xtile) * 256
                        py = (yf - min_ytile) * 256
                        points.append((int(px * scale_x), int(py * scale_y)))
                    if len(points) >= 2:
                        if is_in_path:
                            # Draw soft glow matching congestion color behind active path
                            pygame.draw.lines(screen, (color[0], color[1], color[2]), False, points, thickness + 4)
                        pygame.draw.lines(screen, color, False, points, thickness)
                else:
                    if is_in_path:
                        pygame.draw.line(screen, color, p1, p2, thickness + 4)
                    pygame.draw.line(screen, color, p1, p2, thickness)
                    
        # Draw Moving Traffic Particles
        for p in particles:
            seg_idx = p["segment"]
            progress = p["progress"]
            u_id = current_path[seg_idx]
            v_id = current_path[seg_idx+1]
            
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
            elif current_path and node_id in current_path:
                color = COLOR_PATH
                radius = 8
            else:
                color = COLOR_NODE
                radius = 6
                
            if node_id == hovered_node:
                pygame.draw.circle(screen, COLOR_HOVER, (px, py), radius + 3, 2)
                
            pygame.draw.circle(screen, color, (px, py), radius)
            
            lbl = font_small.render(str(node_id), True, COLOR_TEXT_PRIMARY)
            screen.blit(lbl, (px + 12, py - 7))
            
        # Render Glassmorphism Side Panel
        panel_rect = pygame.Rect(map_w, 0, panel_w, w_h)
        pygame.draw.rect(screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(screen, (60, 60, 75), (map_w, 0), (map_w, w_h), 2)
        
        y_offset = int(w_h * 0.03)
        
        title_surf = font_title.render("TBRGS NAVIGATOR", True, COLOR_ACCENT)
        screen.blit(title_surf, (map_w + 20, y_offset))
        y_offset += int(w_h * 0.07)
        
        # Source/Destination Card
        card_rect = pygame.Rect(map_w + 15, y_offset, panel_w - 30, int(w_h * 0.16))
        pygame.draw.rect(screen, COLOR_CARD_BG, card_rect, border_radius=8)
        pygame.draw.rect(screen, (60, 60, 75), card_rect, 1, border_radius=8)
        
        lbl_s = font_body.render(f"Source ID : {current_start}", True, COLOR_SOURCE)
        lbl_d = font_body.render(f"Dest ID   : {current_dest}", True, COLOR_DEST)
        screen.blit(lbl_s, (map_w + 30, y_offset + 15))
        screen.blit(lbl_d, (map_w + 30, y_offset + 15 + int(w_h * 0.04)))
        
        if current_start in visual_coords:
            c_lat, c_lng = visual_coords[current_start]
            lbl_coords = font_small.render(f"Lat: {c_lat:.5f}, Lng: {c_lng:.5f}", True, COLOR_TEXT_SECONDARY)
            screen.blit(lbl_coords, (map_w + 30, y_offset + 15 + int(w_h * 0.08)))
            
        y_offset += int(w_h * 0.19)
        
        # Route Statistics Card
        stats_header = font_header.render("Route Analytics", True, COLOR_ACCENT)
        screen.blit(stats_header, (map_w + 20, y_offset))
        y_offset += int(w_h * 0.04)
        
        card_stats = pygame.Rect(map_w + 15, y_offset, panel_w - 30, int(w_h * 0.28))
        pygame.draw.rect(screen, COLOR_CARD_BG, card_stats, border_radius=8)
        pygame.draw.rect(screen, (60, 60, 75), card_stats, 1, border_radius=8)
        
        if current_path:
            mins = int(current_travel_time // 60)
            secs = int(current_travel_time % 60)
            time_result = font_body.render(f"Est. Travel Time: {mins} min {secs} s", True, COLOR_SOURCE)
            nodes_count = font_body.render(f"Intersections: {len(current_path)} nodes", True, COLOR_TEXT_PRIMARY)
            
            screen.blit(time_result, (map_w + 30, y_offset + 15))
            screen.blit(nodes_count, (map_w + 30, y_offset + 15 + int(w_h * 0.04)))
            
            path_str = " -> ".join(map(str, current_path))
            words = path_str.split(" -> ")
            lines = []
            curr_line = ""
            for word in words:
                test_line = curr_line + (" -> " if curr_line else "") + word
                if font_small.size(test_line)[0] < panel_w - 60:
                    curr_line = test_line
                else:
                    lines.append(curr_line)
                    curr_line = word
            if curr_line:
                lines.append(curr_line)
                
            path_y = y_offset + 15 + int(w_h * 0.09)
            for idx, line in enumerate(lines[:5]):
                line_surf = font_small.render(line, True, COLOR_TEXT_SECONDARY)
                screen.blit(line_surf, (map_w + 30, path_y))
                path_y += int(w_h * 0.03)
        else:
            no_path = font_body.render("No path exists!", True, COLOR_DEST)
            screen.blit(no_path, (map_w + 30, y_offset + 15))
            
        y_offset += int(w_h * 0.31)
        
        # Interactive Controls Reference
        help_header = font_header.render("Controls Menu", True, COLOR_TEXT_SECONDARY)
        screen.blit(help_header, (map_w + 20, y_offset))
        y_offset += int(w_h * 0.04)
        
        helps = [
            "Left-Click node : Set Source",
            "Right-Click node: Set Destination",
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
