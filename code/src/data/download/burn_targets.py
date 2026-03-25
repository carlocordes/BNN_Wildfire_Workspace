# Internal

# External
import glob
from pathlib import Path
import pandas as pd
import geopandas as gpd
import duckdb

def ingest_burn_records(burn_config) -> None:
    table_name = burn_config['table_name']
    db_name = burn_config['db_name']
    path_to_csv = Path(burn_config['csv'])


    # DB config
    con = duckdb.connect(path_to_csv / db_name)
    con.execute("INSTALL spatial;"
                "LOAD spatial;")
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    first = True

    # csv file schema
    paths = glob.glob(str(path_to_csv) + '/*.csv')


    for load_path in paths:

        # Load and disregard
        points = gpd.read_file(load_path).\
            drop(columns = ["brightness", "scan", "track", "acq_time","satellite",
                            "instrument", "version", "frp", "daynight", "type"])
        
        # Some data wrangling
        points["acq_date"] = pd.to_datetime(points["acq_date"])
        points["year"] = points["acq_date"].dt.year
        points["month"] = points["acq_date"].dt.month
        points["day_of_year"] = points["acq_date"].dt.dayofyear


        points["geometry"] = gpd.points_from_xy(points['longitude'], points['latitude'])
        points = points.drop(columns = ['latitude', 'longitude'])
        points = gpd.GeoDataFrame(
            points,
            crs="EPSG:4326",
        )

        points["geom_wkb"] = points.geometry.to_wkb()
        points = points.drop(columns = "geometry")


        # Write to db
        view_name = 'gdf_view'
        con.register(view_name, points)
        rel = con.from_df(points)

        if first:
            # Create Mode
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            con.execute(
                f"""
                CREATE TABLE {table_name} AS
                SELECT
                    * EXCLUDE geom_wkb,
                    ST_GeomFromWKB(geom_wkb) AS geom
                FROM {view_name};
                """
            )

            first = False
        else:
            # Append mode
            con.execute(
                f"""
                INSERT INTO {table_name}
                SELECT
                    * EXCLUDE geom_wkb,
                    ST_GeomFromWKB(geom_wkb) AS geom
                FROM {view_name};
                """
            )

        
    count = con.execute(f"""
            SELECT COUNT(*) FROM {table_name}
            """).fetchall()[0][0]
    print(f'Wrote {count} to {table_name}')
