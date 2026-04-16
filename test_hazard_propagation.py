import pandas as pd
import geopandas as gpd

from grid import Grid
from grid_mapper import GridMapper
from hazard_injection import HazardInjector
from hazard_propagation import HazardPropagation

# -----------------------------
# Load spatial grid
# -----------------------------
grid = Grid("env/generated_grid.shp")

# -----------------------------
# Load catchments
# -----------------------------
catchments = gpd.read_file("env/all_catchments.shp")

# -----------------------------
# Map catchments to grid
# -----------------------------
mapper = GridMapper(grid.cells)
mapper.map_catchments_to_grid(catchments)

# -----------------------------
# Load flood events
# -----------------------------
events = pd.read_csv("env/datasets/indofloods/floodevents_indofloods.csv")

# -----------------------------
# Inject hazard
# -----------------------------
injector = HazardInjector(grid.cells)
injector.inject_from_events(events)

# -----------------------------
# Run propagation
# -----------------------------
propagator = HazardPropagation(grid.cells)
propagator.propagate()

print("Hazard propagation test complete")