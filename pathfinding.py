import networkx as nx

def route_on_road_network(graph, source_node, target_node, flood_depth, node_to_rc, depth_threshold=0.2):
    """
    Optimized A* pathfinding precisely matching physical OSMnx road networks.
    Dynamically adjusts edge weights to infinity if the pixel underneath the road is flooded.
    """
    # OSMnx drive graphs are MultiDiGraphs. The weight function signature: weight(u, v, edge_dict)
    def dynamic_weight(u, v, edge_data):
        # Extract base physical length of the road
        if 0 in edge_data:
            base_length = edge_data[0].get('length', 10.0)
        else:
            base_length = edge_data.get('length', 10.0)
        
        # Check flood constraint mapping to pixel
        if v in node_to_rc:
            r, c = node_to_rc[v]
            if flood_depth[r, c] > depth_threshold:
                return float('inf') # Impassable road segment!
                
        return base_length
        
    try:
        path = nx.astar_path(graph, source_node, target_node, weight=dynamic_weight)
        return path
    except Exception:
        # No path available (completely blocked or isolated)
        return []
