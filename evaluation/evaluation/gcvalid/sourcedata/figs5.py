
import argparse
import pathlib

import pandas as pd

from gcvalid.sourcedata import data_gauges
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def data_gauges_max(source, fes):
    df = data_gauges(source, fes)[
        ["record_id", "reference", "model", "max_sim", "max_obs"]
    ]
    df_geoclaw = (
        df.loc[df["model"] == "geoclaw", ["record_id", "max_sim"]].copy()
        .rename(columns={"max_sim": "max_geoclaw"})
    )
    df_gesla3 = (
        df[df["reference"] == "gesla3"]
        .groupby("record_id")["max_obs"]
        .first()
        .reset_index()
        .rename(columns={"max_obs": "max_gesla3"})
    )
    df_codec = pd.concat([
        df.loc[df["model"] == "codec", ["record_id", "max_sim"]]
        .rename(columns={"max_sim": "max_codec"}),
        df.loc[df["reference"] == "codec", ["record_id", "max_obs"]]
        .rename(columns={"max_obs": "max_codec"}),
    ])
    return (
        df_geoclaw
        .merge(df_codec, on="record_id", how="outer")
        .merge(df_gesla3, on="record_id", how="outer")
    )


def main():
    parser = argparse.ArgumentParser(description='Plot and summarize results.')
    parser.add_argument(
        '--source', type=str, metavar="SOURCE", default="all",
        choices=['all', 'dfo', 'gfd', 'rapid'],
        help='The flood map source.',
    )
    parser.add_argument(
        '--fes', type=str, metavar="FES_SETTING", default="max",
        choices=["no", "min", "mean", "max"], help='The FES2014 setting used in the GC runs.',
    )
    args = parser.parse_args()

    df = data_gauges_max(args.source, args.fes)
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
