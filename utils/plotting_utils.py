import geopandas as gpd
import utils.data as Data

def get_sgMainland_bbox():
    planningArea_shp = Data.import_planning_area()
    planningArea_shp_mainland = planningArea_shp[~planningArea_shp["PLN_AREA_N"].str.contains("ISLAND")]
    minx = planningArea_shp_mainland.bounds["minx"].min()
    miny = planningArea_shp_mainland.bounds["miny"].min()
    maxx = planningArea_shp_mainland.bounds["maxx"].max()
    maxy = planningArea_shp_mainland.bounds["maxy"].max()

    bbox = (minx, miny, maxx - 4e-4*maxx, maxy)
    return planningArea_shp_mainland, bbox