import geopandas as gpd

from grid import Grid
from grid_mapper import GridMapper

# Load datasets
grid_file = "env/generated_grid.shp"
catchment_file = "env/all_catchments.shp"

catchments = gpd.read_file(catchment_file)

# Initialize grid
grid = Grid(grid_file)

# Create mapper
mapper = GridMapper(grid.cells)

# Map catchments
mapper.map_catchments_to_grid(catchments)

print("Mapping complete")