# visualize.py
# This script loads map.txt directly and displays the street network and A* route in Pygame.
# Usage: python visualize.py

import pygame
import os
import sys
import math
from src.routing.a_star import a_star_search

# Node ID to pixel coordinates (calibrated from map.png)
NODE_PIXELS = {
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
COLOR_BG = (18, 18, 18)
COLOR_PANEL_BG = (30, 30, 35)
COLOR_TEXT_PRIMARY = (240, 240, 240)
COLOR_TEXT_SECONDARY = (180, 180, 180)
COLOR_ACCENT = (0, 229, 255) # Cyan
COLOR_SOURCE = (57, 255, 20) # Bright Lime Green
COLOR_DEST = (255, 7, 58) # Neon Red
COLOR_PATH = (255, 207, 0) # Gold
COLOR_EDGE = (100, 100, 100)
COLOR_NODE = (80, 80, 80)
COLOR_HOVER = (255, 255, 255)

# Simple Graph structure representing loaded data
class GraphNode:
    def __init__(self, node_id, lat, lng):
        self.id = node_id
        self.lat = lat
        self.lng = lng
        self.neighbors = [] # list of (neighbor_id, cost)

class MapGraph:
    def __init__(self):
        self.nodes = {} # node_id -> GraphNode

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
    
    # Parse nodes and edges
    for line in lines[2:]:
        if ":" in line:
            # Parse Node: id:(lat,lng)
            parts = line.split(":")
            nid = int(parts[0])
            coords_str = parts[1].replace("(", "").replace(")", "")
            lat_str, lng_str = coords_str.split(",")
            graph.nodes[nid] = GraphNode(nid, float(lat_str), float(lng_str))
        elif "," in line:
            # Parse Edge: u,v,cost
            parts = line.split(",")
            u = int(parts[0])
            v = int(parts[1])
            cost = float(parts[2])
            if u in graph.nodes and v in graph.nodes:
                graph.nodes[u].neighbors.append((v, cost))
                
    return graph, start_node, dest_node

def save_map_txt(graph, start_node, dest_node, file_path="map.txt"):
    try:
        with open(file_path, "w") as f:
            f.write(f"{start_node}\n")
            f.write(f"{dest_node}\n")
            for nid in sorted(graph.nodes.keys()):
                node = graph.nodes[nid]
                f.write(f"{node.id}:({node.lat:.6f},{node.lng:.6f})\n")
            
            # Write edges
            edges_written = set()
            for u in sorted(graph.nodes.keys()):
                node = graph.nodes[u]
                for v, cost in sorted(node.neighbors, key=lambda x: x[0]):
                    edge_key = (u, v)
                    edges_written.add(edge_key)
                    f.write(f"{u},{v},{int(round(cost))}\n")
    except Exception as e:
        print(f"Warning: Failed to save changes back to map.txt: {e}")

def main():
    graph, start_node, dest_node = load_map_txt("map.txt")
    
    pygame.init()
    pygame.font.init()
    
    try:
        font_title = pygame.font.SysFont("Helvetica", 24, bold=True)
        font_header = pygame.font.SysFont("Helvetica", 18, bold=True)
        font_body = pygame.font.SysFont("Helvetica", 14)
        font_small = pygame.font.SysFont("Helvetica", 11)
    except:
        font_title = pygame.font.Font(None, 28)
        font_header = pygame.font.Font(None, 22)
        font_body = pygame.font.Font(None, 16)
        font_small = pygame.font.Font(None, 13)

    map_w, map_h = 479, 447
    panel_w = 321
    window_w = map_w + panel_w
    window_h = 500
    
    try:
        screen = pygame.display.set_mode((window_w, window_h))
    except pygame.error as e:
        print(f"Warning: Could not launch Pygame GUI window (no display available): {e}")
        print("Note: 'map.txt' contents remain unchanged.")
        return
        
    pygame.display.set_caption("TBRGS Map Visualizer")
    clock = pygame.time.Clock()
    
    # Load map background
    map_image = None
    if os.path.exists("map.png"):
        try:
            map_image = pygame.image.load("map.png").convert()
            map_image = pygame.transform.scale(map_image, (map_w, map_h))
        except Exception as e:
            print(f"Error loading map.png: {e}")
            
    current_start = start_node
    current_dest = dest_node
    current_path = None
    current_travel_time = 0.0
    
    def update_route():
        nonlocal current_path, current_travel_time
        current_path, current_travel_time = a_star_search(graph, current_start, current_dest)
        # Save updated source/destination back to map.txt
        save_map_txt(graph, current_start, current_dest)

    update_route()
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        hovered_node = None
        hovered_edge = None
        
        # Check node hovers
        for node_id, (px, py) in NODE_PIXELS.items():
            dist = math.hypot(mouse_pos[0] - px, mouse_pos[1] - py)
            if dist <= 12:
                hovered_node = node_id
                break
                
        # Check edge hovers if not hovering over a node
        if hovered_node is None and mouse_pos[0] < map_w and mouse_pos[1] < map_h:
            min_dist_to_edge = 8.0
            for node_id, node in graph.nodes.items():
                p1 = NODE_PIXELS.get(node_id)
                if not p1: continue
                for neighbor_id, cost in node.neighbors:
                    p2 = NODE_PIXELS.get(neighbor_id)
                    if not p2: continue
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
                        
        screen.fill(COLOR_BG)
        
        # Draw map background
        if map_image:
            screen.blit(map_image, (0, 0))
        else:
            pygame.draw.rect(screen, (40, 40, 40), (0, 0, map_w, map_h))
            
        # Draw edges
        drawn_edges = set()
        for node_id, node in graph.nodes.items():
            p1 = NODE_PIXELS.get(node_id)
            if not p1: continue
            for neighbor_id, cost in node.neighbors:
                edge_key = tuple(sorted((node_id, neighbor_id)))
                if edge_key in drawn_edges: continue
                drawn_edges.add(edge_key)
                
                p2 = NODE_PIXELS.get(neighbor_id)
                if not p2: continue
                
                # Check path
                is_in_path = False
                if current_path:
                    for idx in range(len(current_path) - 1):
                        if (current_path[idx] == node_id and current_path[idx+1] == neighbor_id) or \
                           (current_path[idx] == neighbor_id and current_path[idx+1] == node_id):
                            is_in_path = True
                            break
                if is_in_path:
                    pygame.draw.line(screen, COLOR_PATH, p1, p2, 5)
                else:
                    pygame.draw.line(screen, COLOR_EDGE, p1, p2, 2)
                    
        # Draw Nodes
        for node_id, (px, py) in NODE_PIXELS.items():
            if node_id == current_start:
                color = COLOR_SOURCE
                radius = 9
            elif node_id == current_dest:
                color = COLOR_DEST
                radius = 9
            elif current_path and node_id in current_path:
                color = COLOR_PATH
                radius = 7
            else:
                color = COLOR_NODE
                radius = 5
                
            if node_id == hovered_node:
                pygame.draw.circle(screen, COLOR_HOVER, (px, py), radius + 3, 2)
                
            pygame.draw.circle(screen, color, (px, py), radius)
            
            lbl = font_small.render(str(node_id), True, COLOR_TEXT_PRIMARY)
            screen.blit(lbl, (px + 10, py - 6))
            
        # Side Panel
        panel_rect = pygame.Rect(map_w, 0, panel_w, window_h)
        pygame.draw.rect(screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(screen, COLOR_NODE, (map_w, 0), (map_w, window_h), 2)
        
        y_offset = 20
        title_surf = font_title.render("TBRGS Map Viewer", True, COLOR_ACCENT)
        screen.blit(title_surf, (map_w + 20, y_offset))
        y_offset += 45
        
        orig_lbl = font_body.render(f"Source Node: {current_start}", True, COLOR_TEXT_PRIMARY)
        dest_lbl = font_body.render(f"Dest Node: {current_dest}", True, COLOR_TEXT_PRIMARY)
        screen.blit(orig_lbl, (map_w + 20, y_offset))
        y_offset += 20
        screen.blit(dest_lbl, (map_w + 20, y_offset))
        y_offset += 35
        
        res_header = font_header.render("A* Route Result", True, COLOR_ACCENT)
        screen.blit(res_header, (map_w + 20, y_offset))
        y_offset += 25
        
        if current_path:
            mins = int(current_travel_time // 60)
            secs = int(current_travel_time % 60)
            time_result = font_body.render(f"Travel Time: {mins} min {secs} s", True, COLOR_SOURCE)
            screen.blit(time_result, (map_w + 20, y_offset))
            y_offset += 25
            
            # Draw path string
            path_str = " -> ".join(map(str, current_path))
            words = path_str.split(" -> ")
            lines = []
            curr_line = ""
            for word in words:
                test_line = curr_line + (" -> " if curr_line else "") + word
                if font_body.size(test_line)[0] < panel_w - 40:
                    curr_line = test_line
                else:
                    lines.append(curr_line)
                    curr_line = word
            if curr_line:
                lines.append(curr_line)
                
            for idx, line in enumerate(lines[:4]):
                line_surf = font_body.render(line, True, COLOR_TEXT_PRIMARY)
                screen.blit(line_surf, (map_w + 20, y_offset))
                y_offset += 20
        else:
            no_path = font_body.render("No path exists!", True, COLOR_DEST)
            screen.blit(no_path, (map_w + 20, y_offset))
            y_offset += 20
            
        # Controls panel
        y_offset = 360
        help_header = font_header.render("Controls", True, COLOR_TEXT_SECONDARY)
        screen.blit(help_header, (map_w + 20, y_offset))
        y_offset += 25
        
        helps = [
            "Left-Click node : Set Source",
            "Right-Click node: Set Destination",
            "Saves start/dest back to map.txt"
        ]
        for h_text in helps:
            help_surf = font_body.render(h_text, True, COLOR_TEXT_SECONDARY)
            screen.blit(help_surf, (map_w + 20, y_offset))
            y_offset += 20
            
        # Tooltips
        if hovered_node is not None:
            node = graph.nodes[hovered_node]
            tooltip_rect = pygame.Rect(mouse_pos[0] + 15, mouse_pos[1] - 35, 180, 50)
            pygame.draw.rect(screen, (20, 20, 20), tooltip_rect)
            pygame.draw.rect(screen, COLOR_ACCENT, tooltip_rect, 1)
            t1 = font_small.render(f"Intersection: {hovered_node}", True, COLOR_TEXT_PRIMARY)
            t2 = font_small.render(f"Lat: {node.lat:.5f}, Lng: {node.lng:.5f}", True, COLOR_TEXT_SECONDARY)
            screen.blit(t1, (mouse_pos[0] + 20, mouse_pos[1] - 30))
            screen.blit(t2, (mouse_pos[0] + 20, mouse_pos[1] - 15))
            
        elif hovered_edge is not None:
            u, v, cost = hovered_edge
            tooltip_rect = pygame.Rect(mouse_pos[0] + 15, mouse_pos[1] - 30, 180, 45)
            pygame.draw.rect(screen, (20, 20, 20), tooltip_rect)
            pygame.draw.rect(screen, COLOR_ACCENT, tooltip_rect, 1)
            t1 = font_small.render(f"Link: {u} -> {v}", True, COLOR_TEXT_PRIMARY)
            t2 = font_small.render(f"Cost: {int(round(cost))} seconds", True, COLOR_TEXT_SECONDARY)
            screen.blit(t1, (mouse_pos[0] + 20, mouse_pos[1] - 25))
            screen.blit(t2, (mouse_pos[0] + 20, mouse_pos[1] - 13))

        pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()

if __name__ == "__main__":
    main()
