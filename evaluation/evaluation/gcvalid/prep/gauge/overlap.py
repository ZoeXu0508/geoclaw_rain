"""
Check which gauge stations are not in GESLA3 (identify with 0.1 degrees tolerance)
"""
import argparse
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd

import gcvalid.util.constants as u_const
import gcvalid.util.io as u_io


TOLERANCE_DEG = 0.1
"""Buffer (in degrees) around gesla3 stations"""


def main():
    parser = argparse.ArgumentParser(description='Check gauge stations not in GESLA3.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    compare_df = u_io.read_compare_df(
        source, "without", "aviso-fes_max", 0.0, apply_filters=True, verbose=True,
    )

    reference_maps = compare_df['map_id'].values

    missing = []
    maps_without_tg_data = []
    maps_without_usable_tg_data = []
    maps_without_g3_data = []
    for ifile, map_id in enumerate(reference_maps):
        print(f"Processing file {ifile:3d}/{len(reference_maps)} ...", end="\r", flush=True)

        gfile = u_const.GAUGES_DIR / source / "records" / f"{map_id}.pickle"
        with gfile.open("rb") as fp:
            gdata = pickle.load(fp)

        all_stations_w_discarded = pd.DataFrame([
            {
                "mapid": map_id,
                "gsrc": gsrc,
                "name": d['filename'],
                "lat": d['location'][0],
                "lon": d['location'][1],
                "discarded": "" if d['discarded'] == False else d['discarded'],
            }
            for gsrc, stdata in gdata.items() if gsrc != "gtsm"
            for d in stdata
        ])

        if all_stations_w_discarded.size == 0:
            maps_without_tg_data.append(map_id)
            continue

        all_stations = all_stations_w_discarded[all_stations_w_discarded['discarded'] == ""]
        all_stations = all_stations.drop(axis=1, columns=["discarded"])

        if all_stations.size == 0:
            maps_without_usable_tg_data.append(map_id)
            continue

        all_stations = gpd.GeoDataFrame(
            all_stations,
            geometry=gpd.points_from_xy(all_stations["lon"], all_stations["lat"]))

        gesla3_w_buffer = all_stations[all_stations['gsrc'] == 'gesla3'].copy()
        gesla3_w_buffer['geometry'] = gesla3_w_buffer.geometry.buffer(TOLERANCE_DEG)

        if gesla3_w_buffer.size == 0:
            maps_without_g3_data.append(map_id)

        all_stations['gesla3'] = np.nan
        for idx, row in gesla3_w_buffer.iterrows():
            all_stations.loc[all_stations.within(row.geometry), "gesla3"] = row["name"]

        missing.append(all_stations[all_stations["gesla3"].isna()])
    print(f"Processing file {len(reference_maps)}/{len(reference_maps)} ...")

    print(f"No data ({len(maps_without_tg_data)}): {', '.join(maps_without_tg_data)}")
    print(f"Only discarded data ({len(maps_without_usable_tg_data)}): {', '.join(maps_without_usable_tg_data)}")
    print(f"No GESLA3 data ({len(maps_without_g3_data)}): {', '.join(maps_without_g3_data)}")

    missing = pd.concat(missing).groupby(by=["gsrc", "name"]).first().reset_index()
    print(f"For {np.unique(missing[missing['gsrc'] == 'wsl'].mapid).size} maps, WSL has data that is not in GESLA3.")

    print(missing)


if __name__ == "__main__":
    main()
