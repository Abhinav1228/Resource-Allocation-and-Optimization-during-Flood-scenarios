import pandas as pd
import geopandas as gpd

from grid import Grid
from grid_mapper import GridMapper
from hazard_injection import HazardInjector

grid = Grid("env/generated_grid.shp")

catchments = gpd.read_file("env/all_catchments.shp")

mapper = GridMapper(grid.cells)
mapper.map_catchments_to_grid(catchments)

events = pd.read_csv("env/datasets/indofloods/floodevents_indofloods.csv")
print(events.columns)
injector = HazardInjector(grid.cells)

injector.inject_from_events(events)

print("Hazard injection complete")