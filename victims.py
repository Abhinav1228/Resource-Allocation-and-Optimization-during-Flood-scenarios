import numpy as np

class Incident:
    def __init__(self, inc_id, r, c, severity=1.0):
        self.id = inc_id
        self.r = r
        self.c = c
        self.severity = severity
        self.node_id = None
        
        self.risk_level = 0.0
        self.is_resolved = False
        self.assigned_unit = None

    def tick_risk(self, flood_depth_at_location):
        if not self.is_resolved:
            if flood_depth_at_location > 0.05:
                self.risk_level += (flood_depth_at_location * 0.1 * self.severity)
            self.risk_level = min(1.0, self.risk_level)

    def resolve(self):
        self.is_resolved = True
        self.risk_level = 0.0
        
class IncidentManager:
    def __init__(self, rem):
        self.incidents = []
        self.rem = rem
        self.id_counter = 0
        
    def spawn_strategic_incidents(self, count):
        """
        Spawns victims strategically:
        - 40% in the lowest-elevation areas (flood path) to show rerouting
        - 60% in medium-elevation areas (safe but nearby) to show dispatch
        """
        H, W = self.rem.shape
        
        # Identify flood-prone zones (lowest 20% of REM)
        flat_rem = self.rem.flatten()
        threshold_low = np.percentile(flat_rem[flat_rem > 0], 20)
        threshold_mid = np.percentile(flat_rem[flat_rem > 0], 60)
        
        # Flood-zone victims (40% of total)
        flood_count = max(1, int(count * 0.4))
        safe_count = count - flood_count
        
        spawned = 0
        attempts = 0
        while spawned < flood_count and attempts < 5000:
            r = np.random.randint(0, H)
            c = np.random.randint(0, W)
            attempts += 1
            # Low elevation = flood-prone
            if 0 < self.rem[r, c] <= threshold_low:
                inc = Incident(self.id_counter, r, c, severity=1.5)  # Higher severity
                self.incidents.append(inc)
                self.id_counter += 1
                spawned += 1
        
        # Nearby safe-zone victims (60% of total)
        spawned = 0
        attempts = 0
        while spawned < safe_count and attempts < 5000:
            r = np.random.randint(0, H)
            c = np.random.randint(0, W)
            attempts += 1
            if threshold_low < self.rem[r, c] <= threshold_mid:
                inc = Incident(self.id_counter, r, c, severity=1.0)
                self.incidents.append(inc)
                self.id_counter += 1
                spawned += 1
                
    def spawn_random_incidents(self, count):
        """Fallback: Random placement on non-zero terrain."""
        self.spawn_strategic_incidents(count)
                
    def get_active_incidents(self):
        return [inc for inc in self.incidents if not inc.is_resolved]
        
    def update_risks(self, flood_depth_matrix):
        for inc in self.incidents:
            inc.tick_risk(flood_depth_matrix[inc.r, inc.c])
