import numpy as np

class RescueUnit:
    def __init__(self, unit_id, r, c, unit_type="Ambulance"):
        self.id = unit_id
        self.r = r
        self.c = c
        self.node_id = None
        self.type = unit_type
        
        self.status = "idle" # idle, en-route, busy
        self.target_incident = None
        self.path_nodes = [] # List of OSMnx nodes to travel
        
        # Heuristic capacity or speed
        self.speed = 2 if unit_type == "Ambulance" else 1

    def assign_task(self, incident, path_nodes):
        """Dispatches unit to a target via calculated node path."""
        self.target_incident = incident
        self.path_nodes = path_nodes
        self.status = "en-route"

    def step_move(self, node_to_rc):
        """Moves the unit along the path based on its speed."""
        if self.status != "en-route" or not self.path_nodes:
            return False # Didn't move
            
        # Move up to `self.speed` steps along the path
        steps_taken = 0
        while self.path_nodes and steps_taken < self.speed:
            next_node = self.path_nodes.pop(0)
            self.node_id = next_node
            self.r, self.c = node_to_rc[next_node]
            steps_taken += 1

        # Check arrival
        if not self.path_nodes:
            self.status = "busy" # Reached target, now resolving
            return True # Reached destination
            
        return False
        
    def resolve_task(self):
        """Completes the assignment and reverts to idle."""
        self.status = "idle"
        self.target_incident = None
        
    def get_position(self):
        return self.r, self.c

class Ambulance(RescueUnit):
    def __init__(self, unit_id, r, c):
        super().__init__(unit_id, r, c, unit_type="Ambulance")

class Firefighter(RescueUnit):
    def __init__(self, unit_id, r, c):
        super().__init__(unit_id, r, c, unit_type="Firefighter")
