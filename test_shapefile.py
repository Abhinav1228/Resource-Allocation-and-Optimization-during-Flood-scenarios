import geopandas as gpd
import os

base_dir = "env/datasets/indofloods/catchments_shapefiles_indofloods"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".shp"):
            shapefile_path = os.path.join(root, file)
            print("Found shapefile:", shapefile_path)

            gdf = gpd.read_file(shapefile_path)

            print("Shapefile loaded successfully!")
            print(gdf.head())
            print("CRS:", gdf.crs)
            print("Geometry type:", gdf.geometry.type)

            break