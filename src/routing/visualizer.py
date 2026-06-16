# src/routing/visualizer.py
# This module implements the Pygame-based map visualization.

import pygame
import sys
import os
import math
from src.routing.a_star import a_star_search
from src.models.registry import get_model
from src.routing.speed_converter import flow_to_speed, calculate_travel_time

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

def run_visualizer(graph, initial_path, start_node_id, dest_node_id, initial_time_str, initial_model_name):
    pygame.init()
    pygame.font.init()
    
    # Fonts
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

    # Window configuration
    map_w, map_h = 479, 447
    panel_w = 321
    window_w = map_w + panel_w
    window_h = 500
    
    try:
        screen = pygame.display.set_mode((window_w, window_h))
    except pygame.error as e:
        print(f"Warning: Could not launch Pygame GUI window (no display available): {e}")
        print("Note: 'map.txt' has been generated successfully in the project directory.")
        return
        
    pygame.display.set_caption("TBRGS - Traffic-Based Route Guidance System")
    clock = pygame.time.Clock()
    
    # Load background map image
    map_image = None
    if os.path.exists("map.png"):
        try:
            map_image = pygame.image.load("map.png").convert()
            # Scale if needed, but it should be 479x447
            map_image = pygame.transform.scale(map_image, (map_w, map_h))
        except Exception as e:
            print(f"Error loading map.png: {e}")
            
    # State variables
    current_start = start_node_id
    current_dest = dest_node_id
    current_model_name = initial_model_name.upper()
    
    # Parse time
    try:
        hour = int(initial_time_str[:2])
        minute = int(initial_time_str[2:])
    except:
        hour = 11
        minute = 0
        
    current_path = initial_path
    current_travel_time = 0.0
    
    def update_graph_and_path():
        nonlocal current_path, current_travel_time
        time_str = f"{hour:02d}{minute:02d}"
        
        # Instantiate the model from the registry
        model_class = get_model(current_model_name)
        model_inst = model_class()
        
        # Reload edge costs dynamically
        graph.load_from_csv(
            lookup_path="data/processed/movement_lookup.csv",
            edges_path="data/processed/edges.csv",
            model_instance=model_inst,
            time_str=time_str
        )
        
        # Write the new map.txt
        # (from A2B.py or main entry: we can write it dynamically too!)
        # Write map.txt as the user wanted
        try:
            with open("map.txt", "w") as f:
                f.write(f"{current_start}\n")
                f.write(f"{current_dest}\n")
                for node_id in sorted(graph.nodes.keys()):
                    node = graph.nodes[node_id]
                    f.write(f"{node.id}:({node.lat:.6f},{node.lng:.6f})\n")
                
                # Write unique edges
                edges_written = set()
                for u in sorted(graph.nodes.keys()):
                    node = graph.nodes[u]
                    for v, cost in sorted(node.neighbors, key=lambda x: x[0]):
                        edge_key = (u, v)
                        edges_written.add(edge_key)
                        f.write(f"{u},{v},{int(round(cost))}\n")
        except Exception as e:
            print(f"Error writing map.txt: {e}")
            
        # Run A* Search
        current_path, current_travel_time = a_star_search(graph, current_start, current_dest)

    # Initial calculation
    update_graph_and_path()
    
    running = True
    while running:
        # 1. Event Handling
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
            # Find closest edge
            min_dist_to_edge = 8.0
            for node_id, node in graph.nodes.items():
                p1 = NODE_PIXELS.get(node_id)
                if not p1: continue
                for neighbor_id, cost in node.neighbors:
                    p2 = NODE_PIXELS.get(neighbor_id)
                    if not p2: continue
                    # Distance from mouse_pos to line segment p1-p2
                    x0, y0 = mouse_pos
                    x1, y1 = p1
                    x2, y2 = p2
                    # Compute segment distance
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
                        update_graph_and_path()
                    elif event.button == 3: # Right click
                        current_dest = hovered_node
                        update_graph_and_path()
                        
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    current_model_name = "LSTM"
                    update_graph_and_path()
                elif event.key == pygame.K_g:
                    current_model_name = "GRU"
                    update_graph_and_path()
                elif event.key == pygame.K_r:
                    current_model_name = "RANDOM"
                    update_graph_and_path()
                elif event.key == pygame.K_UP:
                    hour = (hour + 1) % 24
                    update_graph_and_path()
                elif event.key == pygame.K_DOWN:
                    hour = (hour - 1) % 24
                    update_graph_and_path()
                elif event.key == pygame.K_RIGHT:
                    minute = (minute + 15) % 60
                    update_graph_and_path()
                elif event.key == pygame.K_LEFT:
                    minute = (minute - 15) % 60
                    update_graph_and_path()
                    
        # 2. Rendering
        screen.fill(COLOR_BG)
        
        # Draw Map background
        if map_image:
            screen.blit(map_image, (0, 0))
        else:
            pygame.draw.rect(screen, (40, 40, 40), (0, 0, map_w, map_h))
            
        # Draw Street network edges
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
                
                # Check if this edge is in the current path
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
            # Choose color
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
            
            # Label node ID
            lbl = font_small.render(str(node_id), True, COLOR_TEXT_PRIMARY)
            screen.blit(lbl, (px + 10, py - 6))
            
        # 3. Draw Side Panel
        panel_rect = pygame.Rect(map_w, 0, panel_w, window_h)
        pygame.draw.rect(screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(screen, COLOR_NODE, (map_w, 0), (map_w, window_h), 2)
        
        # Panel Content
        y_offset = 20
        title_surf = font_title.render("TBRGS Navigation", True, COLOR_ACCENT)
        screen.blit(title_surf, (map_w + 20, y_offset))
        y_offset += 40
        
        # Origin and Dest info
        orig_lbl = font_body.render(f"Source: {current_start}", True, COLOR_TEXT_PRIMARY)
        dest_lbl = font_body.render(f"Destination: {current_dest}", True, COLOR_TEXT_PRIMARY)
        screen.blit(orig_lbl, (map_w + 20, y_offset))
        y_offset += 20
        screen.blit(dest_lbl, (map_w + 20, y_offset))
        y_offset += 30
        
        # Settings info
        time_lbl = font_body.render(f"Time: {hour:02d}:{minute:02d} (24-hour)", True, COLOR_TEXT_PRIMARY)
        model_lbl = font_body.render(f"Prediction: {current_model_name}", True, COLOR_TEXT_PRIMARY)
        screen.blit(time_lbl, (map_w + 20, y_offset))
        y_offset += 20
        screen.blit(model_lbl, (map_w + 20, y_offset))
        y_offset += 35
        
        # Path result
        path_header = font_header.render("Routing Results", True, COLOR_ACCENT)
        screen.blit(path_header, (map_w + 20, y_offset))
        y_offset += 25
        
        if current_path:
            # Travel Time formatting
            mins = int(current_travel_time // 60)
            secs = int(current_travel_time % 60)
            time_result = font_body.render(f"Est. Travel Time: {mins} min {secs} s", True, COLOR_SOURCE)
            screen.blit(time_result, (map_w + 20, y_offset))
            y_offset += 25
            
            # Format path string to fit screen width
            path_str = " -> ".join(map(str, current_path))
            # Wrap path string if it is too long
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
                
            for idx, line in enumerate(lines[:4]): # limit to 4 lines to fit panel
                line_surf = font_body.render(line, True, COLOR_TEXT_PRIMARY)
                screen.blit(line_surf, (map_w + 20, y_offset))
                y_offset += 20
        else:
            no_path = font_body.render("No path found!", True, COLOR_DEST)
            screen.blit(no_path, (map_w + 20, y_offset))
            y_offset += 20
            
        y_offset = 340
        # Interactivity Help panel
        help_header = font_header.render("Controls", True, COLOR_TEXT_SECONDARY)
        screen.blit(help_header, (map_w + 20, y_offset))
        y_offset += 20
        
        helps = [
            "L-Click node : Set Source",
            "R-Click node : Set Destination",
            "L / G / R key: Choose Model (LSTM/GRU/RAND)",
            "UP / DOWN key: Change departure hour",
            "LEFT/RIGHT   : Change departure minute"
        ]
        for h_text in helps:
            help_surf = font_small.render(h_text, True, COLOR_TEXT_SECONDARY)
            screen.blit(help_surf, (map_w + 20, y_offset))
            y_offset += 16
            
        # Draw Hover tooltip
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
            tooltip_rect = pygame.Rect(mouse_pos[0] + 15, mouse_pos[1] - 40, 200, 60)
            pygame.draw.rect(screen, (20, 20, 20), tooltip_rect)
            pygame.draw.rect(screen, COLOR_ACCENT, tooltip_rect, 1)
            
            t1 = font_small.render(f"Link: {u} -> {v}", True, COLOR_TEXT_PRIMARY)
            t2 = font_small.render(f"Cost: {int(round(cost))} seconds", True, COLOR_TEXT_SECONDARY)
            # Find the flow value
            model_class = get_model(current_model_name)
            model_inst = model_class()
            time_str = f"{hour:02d}{minute:02d}"
            flow = model_inst.predict(from_site=u, to_site=v, time_str=time_str)
            t3 = font_small.render(f"Flow: {int(flow)} vehicles/hour", True, COLOR_TEXT_SECONDARY)
            
            screen.blit(t1, (mouse_pos[0] + 20, mouse_pos[1] - 35))
            screen.blit(t2, (mouse_pos[0] + 20, mouse_pos[1] - 22))
            screen.blit(t3, (mouse_pos[0] + 20, mouse_pos[1] - 10))

        pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()
