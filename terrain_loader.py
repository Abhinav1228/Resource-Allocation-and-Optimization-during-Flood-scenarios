import rioxarray as rxr
import xarray as xr
import osmnx as ox
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from scipy.spatial import KDTree
import geopandas as gpd

class TerrainLoader:
    def __init__(self, tif_files):
        self.tif_files = tif_files
        self.dem = None
        self.rem = None
        self.transform = None
        self.road_graph = None
        self.node_to_rc = {}
        
    def load_and_crop_dem(self):
        print("Merging DEM files...")
        src_files = [rasterio.open(f) for f in self.tif_files]
        mosaic, transform = merge(src_files)
        
        # Bandra-Kurla / Mithi River basin — Mumbai's most flood-prone coastal zone
        self.min_lon, self.min_lat = 72.84, 19.04
        self.max_lon, self.max_lat = 72.88, 19.08
        
        window = from_bounds(self.min_lon, self.min_lat, self.max_lon, self.max_lat, transform)
        row_start, col_start = int(window.row_off), int(window.col_off)
        height, width = int(window.height), int(window.width)
        
        self.dem = mosaic[0, row_start:row_start + height, col_start:col_start + width]
        self.transform = rasterio.windows.transform(window, transform)
        self.dem = self.dem.astype(np.float32)
        print(f"DEM loaded and cropped. Shape: {self.dem.shape}")

    def download_road_network(self):
        """Downloads OSMnx road graph for the specific bounding box matching the DEM."""
        print("Fetching Mumbai OSMnx Drive Network (This takes a few seconds)...")
        try:
            # osmnx bounds: (west, south, east, north) -> (min_lon, min_lat, max_lon, max_lat)
            self.road_graph = ox.graph_from_bbox(
                bbox=(self.min_lon, self.min_lat, self.max_lon, self.max_lat),
                network_type='drive', 
                simplify=True
            )
            print(f"Road network downloaded! Nodes: {len(self.road_graph.nodes)}")
            
            # Map road nodes back to raster coordinates (row, col)
            for node, data in self.road_graph.nodes(data=True):
                lon, lat = data['x'], data['y']
                row, col = rasterio.transform.rowcol(self.transform, lon, lat)
                
                # Clamp boundaries
                row = max(0, min(self.dem.shape[0]-1, row))
                col = max(0, min(self.dem.shape[1]-1, col))
                
                self.node_to_rc[node] = (row, col)
                
        except Exception as e:
            print(f"Failed to fetch OSMnx road network: {e}")
            self.road_graph = None

    def compute_rem(self, river_name="Ulhas River"):
        """Computes Relative Elevation Model using KDTree interpolation from OSM river."""
        print(f"Fetching '{river_name}' geometry from OSM...")
        try:
            # We'll use a local fallback if this fails since Nominatim blocked the first one.
            river_gdf = ox.features_from_place("Mumbai, India", tags={'waterway': 'river'})
            river_gdf = river_gdf[river_gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
        except Exception as e:
            print("Using fallback dummy REM computation.")
            self.rem = self.dem - np.min(self.dem)
            self.rem = np.maximum(self.rem, 0)
            return self.rem
            
        print("Extracting river coordinates...")
        river_coords = []
        for geom in river_gdf.geometry:
            if geom.geom_type == 'LineString':
                river_coords.extend(list(geom.coords))
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    river_coords.extend(list(line.coords))
                    
        print(f"Sampling {len(river_coords)} river points against DEM...")
        river_elevations = []
        valid_coords = []
        
        for lon, lat in river_coords:
            row, col = rasterio.transform.rowcol(self.transform, lon, lat)
            if 0 <= row < self.dem.shape[0] and 0 <= col < self.dem.shape[1]:
                river_elevations.append(self.dem[row, col])
                valid_coords.append((lon, lat))
                
        if not valid_coords:
            self.rem = self.dem - np.min(self.dem)
            return self.rem
            
        print("Building KDTree for river surface interpolation...")
        tree = KDTree(valid_coords)
        
        H, W = self.dem.shape
        cols, rows = np.meshgrid(np.arange(W), np.arange(H))
        all_lons, all_lats = rasterio.transform.xy(self.transform, rows.flatten(), cols.flatten())
        all_pixel_coords = np.column_stack((all_lons, all_lats))
        
        K = min(10, len(valid_coords))
        distances, indices = tree.query(all_pixel_coords, k=K)
        
        weights = 1.0 / (distances + 1e-8)
        river_elevations_arr = np.array(river_elevations)
        river_surface = np.average(
            river_elevations_arr[indices], weights=weights, axis=1
        ).reshape(self.dem.shape)
        
        self.rem = self.dem - river_surface
        self.rem = np.maximum(self.rem, 0)
        
        return self.rem