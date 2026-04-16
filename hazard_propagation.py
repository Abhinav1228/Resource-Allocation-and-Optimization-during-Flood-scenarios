import heapq
import numpy as np

class HazardPropagation:
    def __init__(self, rem):
        self.rem = rem
        self.H, self.W = rem.shape

    def propagate(self, flood_depth, source_pixels, continuous_inflow_volume=15.0):
        """
        Min-Heap based topographical fluid spillover.
        Optimized to continuously inject volume to create a stark visual wave.
        """
        visited = np.zeros((self.H, self.W), dtype=bool)
        heap = []
        
        # Inject continuous surging water per frame for hyper-visibility
        for r, c in source_pixels:
            flood_depth[r, c] += continuous_inflow_volume
            ws = self.rem[r, c] + flood_depth[r, c]
            heapq.heappush(heap, (ws, r, c))
            visited[r, c] = True
            
        # We manually step the algorithm an exact number of iterations to simulate velocity!
        max_spread_iterations = len(heap) * 1000 # Increased massively for extreme propagation
        iters = 0
        
        while heap and iters < max_spread_iterations:
            current_ws, r, c = heapq.heappop(heap)
            iters += 1
            
            # Flood downhill neighbors
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = r+dr, c+dc
                
                if 0 <= nr < self.H and 0 <= nc < self.W and not visited[nr, nc]:
                    visited[nr, nc] = True
                    neighbor_ws = self.rem[nr, nc] + flood_depth[nr, nc]
                    
                    if neighbor_ws < current_ws:
                        # Massive transfer to simulate surge dynamics
                        flow_amount = (current_ws - neighbor_ws) * 0.95
                        actual_flow = min(flow_amount, flood_depth[r, c])
                        
                        if actual_flow > 0.05:
                            flood_depth[r, c] -= (actual_flow * 0.5)
                            flood_depth[nr, nc] += actual_flow
                            
                            new_ws = self.rem[nr, nc] + flood_depth[nr, nc]
                            heapq.heappush(heap, (new_ws, nr, nc))
                            
        # Removed the heavy `*= 0.95` exponential decay to preserve flood pools for analysis
        flood_depth[flood_depth < 0.01] = 0
        
        return flood_depth