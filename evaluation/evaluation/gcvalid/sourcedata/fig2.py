
import argparse
import pathlib

import pandas as pd

from gcvalid.compare.hwm import read_hwms
import gcvalid.util.constants as u_const
import gcvalid.util.gauge as u_gauge
import gcvalid.util.io as u_io


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def data_extents(source):
    df = pd.concat([
        u_io.read_compare_df(
            src, "without", "aviso-fes_max", 0.1,
            apply_filters=True, verbose=True,
        )
        for src in (['gfd', 'dfo', 'rapid'] if source == "all" else [source])
    ])
    df["map_id"] = df["source"] + "_" + df["map_id"]
    return df[['map_id', 'lon_min', 'lat_min', 'lon_max', 'lat_max', 'maxwind']]


def data_gauges(source):
    return (
        pd.concat([
            u_gauge.load_gauge_locations(src, "gesla3")
            for src in (['gfd', 'dfo', 'rapid'] if source == "all" else [source])
        ])
        .rename(columns={"name": "gauge_id"})
        .groupby(by=["gauge_id"])
        .first()
        .reset_index()
        [["gauge_id", "lat", "lon"]]
    )


def data_hwms(source):
    df = (
        pd.concat([
            read_hwms(src, add_flooded_status=False, as_dataframe=True)
            for src in (['gfd', 'dfo', 'rapid'] if source == "all" else [source])
        ])
        .rename(columns={"latitude": "lat", "longitude": "lon"})
        .groupby(by=["ibtracs_id", "hwm_id"])
        .first()
        .reset_index()
    )
    df = df[df["hwm_environment"] == "Coastal"].copy()
    df["hwm_id"] = df["ibtracs_id"] + "_" + df["hwm_id"].astype(str)
    return df[["hwm_id", "lat", "lon"]]


def main():
    parser = argparse.ArgumentParser(description='Plot and summarize observational data.')
    parser.add_argument(
        '--source', type=str, metavar="SOURCE", default="all",
        choices=['all', 'dfo', 'gfd', 'rapid'],
        help='The flood map source.',
    )
    args = parser.parse_args()

    df = data_extents(args.source)
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-extents.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)

    df = data_gauges(args.source)
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-gauges.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)

    df = data_hwms(args.source)
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-hwms.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
