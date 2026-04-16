import geopandas as gpd
import pandas as pd
import os

base_dir = "env/datasets/indofloods/catchments_shapefiles_indofloods"

catchments = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".shp"):
            path = os.path.join(root, file)

            gdf = gpd.read_file(path)

            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")

            gdf = gdf.to_crs("EPSG:4326")

            catchments.append(gdf)

all_catchments = gpd.GeoDataFrame(
    pd.concat(catchments, ignore_index=True),
    crs="EPSG:4326"
)

bounds = all_catchments.total_bounds

print("Dataset bounds:")
print("Min Longitude:", bounds[0])
print("Min Latitude:", bounds[1])
print("Max Longitude:", bounds[2])
print("Max Latitude:", bounds[3])