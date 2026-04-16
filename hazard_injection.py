import numpy as np
import rasterio

class HazardInjector:
    def __init__(self, transform, grid_shape):
        self.transform = transform
        self.grid_shape = grid_shape

    def inject_from_events(self, flood_events):
        """Maps Lat/Lon flood events to raster (row, col) pixel indices."""
        source_pixels = []
        seen = set()
        
        for _, row in flood_events.iterrows():
            lat = row.get('Latitude', None)
            lon = row.get('Longitude', None)
            level = row.get('Peak Flood Level (m)', 5.0)
            
            if lat is None or lon is None:
                continue
                
            r, c = rasterio.transform.rowcol(self.transform, lon, lat)
            if 0 <= r < self.grid_shape[0] and 0 <= c < self.grid_shape[1]:
                key = (r, c)
                if key not in seen:
                    seen.add(key)
                    source_pixels.append((r, c, float(level)))
                    
        print(f"Hazard injected into {len(source_pixels)} unique grid cells.")
        return source_pixels
    
    @staticmethod
    def find_coastal_sources(rem, num_sources=5):
        """
        Finds the lowest-elevation pixels as natural flood injection points.
        These represent river mouths, coastline, and low-lying basins.
        """
        flat = rem.flatten()
        # Get indices of the lowest non-zero elevation cells
        valid_mask = flat > 0
        if not valid_mask.any():
            # All zero — just pick corners
            H, W = rem.shape
            return [(0, 0, 20.0)]
        
        valid_indices = np.where(valid_mask)[0]
        valid_elevations = flat[valid_indices]
        
        # Pick the N lowest
        sorted_idx = np.argsort(valid_elevations)[:num_sources * 10]
        
        # Spread them out spatially (don't cluster all sources together)
        selected = []
        min_dist = min(rem.shape) // 6  # Minimum pixel distance between sources
        
        for idx in sorted_idx:
            global_idx = valid_indices[idx]
            r, c = np.unravel_index(global_idx, rem.shape)
            
            too_close = False
            for sr, sc, _ in selected:
                if abs(r - sr) + abs(c - sc) < min_dist:
                    too_close = True
                    break
            if not too_close:
                selected.append((int(r), int(c), 25.0))
                if len(selected) >= num_sources:
                    break
        
        if not selected:
            selected = [(int(valid_indices[0] // rem.shape[1]), 
                         int(valid_indices[0] % rem.shape[1]), 25.0)]
        
        return selected