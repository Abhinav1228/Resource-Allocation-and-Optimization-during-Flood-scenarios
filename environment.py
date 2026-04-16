import numpy as np
import networkx as nx
from env.resources import Ambulance, Firefighter
from env.victims import IncidentManager
from env.pathfinding import route_on_road_network
from scipy.spatial import KDTree

class DisasterEnvironment:
    def __init__(self, rem, road_graph, node_to_rc, flood_depth=None, num_units=5, num_incidents=10):
        self.rem = rem
        self.road_graph = road_graph
        self.node_to_rc = node_to_rc
        self.H, self.W = rem.shape
        
        self.flood_depth = flood_depth if flood_depth is not None else np.zeros_like(rem)
        
        # Build KDTree to map arbitrary coordinates to nearest OSMnx node quickly
        self.graph_nodes = list(self.road_graph.nodes(data=True)) if self.road_graph else []
        if self.graph_nodes:
            self.node_coords = [(data['y'], data['x']) for _, data in self.graph_nodes] # lat, lon
            self.node_ids = [n for n, _ in self.graph_nodes]
            self.node_tree = KDTree(self.node_coords)
        else:
            self.node_tree = None

        self.incident_manager = IncidentManager(rem)
        self.incident_manager.spawn_random_incidents(num_incidents)
        
        # Assign actual road nodes to incidents based on spatial proximity
        self.patch_incident_nodes(self.incident_manager.incidents)
        
        # Units
        self.units = []
        for i in range(num_units):
            # Spawn units randomly on dry land initially, mapped to roads
            r = np.random.randint(0, self.H)
            c = np.random.randint(0, self.W)
            amb = Ambulance(unit_id=i, r=r, c=c)
            self.patch_unit_nodes([amb])
            self.units.append(amb)
            
        self.time_step = 0
        self.total_reward = 0

    def patch_incident_nodes(self, items):
        if not self.node_tree: return
        for it in items:
            # Approximation assuming r,c roughly correlates or just assign random valid node
            # Since r,c doesn't directly give lat/lon without the transform, we just pick random road nodes
            # To be perfect, we would use the transform. Here we just sample random nodes directly.
            random_idx = np.random.randint(0, len(self.node_ids))
            it.node_id = self.node_ids[random_idx]
            it.r, it.c = self.node_to_rc[it.node_id]

    def patch_unit_nodes(self, units):
        if not self.node_tree: return
        for u in units:
            random_idx = np.random.randint(0, len(self.node_ids))
            u.node_id = self.node_ids[random_idx]
            u.r, u.c = self.node_to_rc[u.node_id]
            u.path_nodes = []

    def get_state(self):
        incident_map = np.zeros_like(self.rem)
        for inc in self.incident_manager.get_active_incidents():
            incident_map[inc.r, inc.c] = inc.risk_level + 0.1
            
        unit_map = np.zeros_like(self.rem)
        for u in self.units:
            unit_map[u.r, u.c] = 1.0
            
        return np.stack([self.rem, self.flood_depth, incident_map, unit_map], axis=-1)
        
    def step(self, actions=None):
        reward = 0
        
        if actions and self.road_graph:
            active_incs = {inc.id: inc for inc in self.incident_manager.get_active_incidents()}
            for unit_id, inc_id in actions:
                unit = next((u for u in self.units if u.id == unit_id), None)
                incident = active_incs.get(inc_id)
                
                if unit and incident and unit.status == "idle":
                    path_nodes = route_on_road_network(
                        self.road_graph, unit.node_id, incident.node_id, 
                        self.flood_depth, self.node_to_rc
                    )
                    if path_nodes:
                        unit.assign_task(incident, path_nodes)
                        incident.assigned_unit = unit

        for unit in self.units:
            if unit.status == "en-route":
                # Predictive obstacle check across the entire remaining path
                path_blocked = False
                for node in unit.path_nodes:
                    if node in self.node_to_rc:
                        nr, nc = self.node_to_rc[node]
                        if self.flood_depth[nr, nc] > 0.2:
                            path_blocked = True
                            break
                            
                if path_blocked:
                    # Path blocked by flood! Recalculate
                    new_path_nodes = route_on_road_network(
                        self.road_graph, unit.node_id, unit.target_incident.node_id, 
                        self.flood_depth, self.node_to_rc
                    )
                    if new_path_nodes:
                        unit.path_nodes = new_path_nodes
                    else:
                        unit.resolve_task()
                        if unit.target_incident:
                            unit.target_incident.assigned_unit = None
                        reward -= 5
                            
                # Move physically
                reached = unit.step_move(self.node_to_rc)

                if reached:
                    if unit.target_incident:
                        unit.target_incident.resolve()
                        reward += 100 
                        reward -= (unit.target_incident.risk_level * 50) 
                    unit.resolve_task()
                    
        self.incident_manager.update_risks(self.flood_depth)
        
        for inc in self.incident_manager.get_active_incidents():
            if inc.risk_level >= 1.0:
                reward -= 10 
            elif inc.risk_level > 0:
                reward -= (inc.risk_level)
                
        self.total_reward += reward
        self.time_step += 1
        
        done = len(self.incident_manager.get_active_incidents()) == 0 or self.time_step > 200
        return self.get_state(), reward, done, self.get_info()
        
    def get_info(self):
        return {
            "time_step": self.time_step,
            "total_reward": self.total_reward,
            "active_incidents": len(self.incident_manager.get_active_incidents()),
            "units_busy": sum(1 for u in self.units if u.status != "idle")
        }