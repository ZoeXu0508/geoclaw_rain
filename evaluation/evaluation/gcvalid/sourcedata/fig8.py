
import argparse
import pathlib

import numpy as np
import pandas as pd

from gcvalid.sourcedata import GAUGE_METRICS, data_gauges
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def print_stats_gauges(df):
    for ref in ["gesla3", "codec"]:
        df_ref = df[(df["model"] == "geoclaw") & (df["reference"] == ref)].copy()
        n_records = df_ref.shape[0]
        n_gauges = np.unique(df_ref["stname"]).size
        n_storms = np.unique(df_ref["ibtracs_id"]).size
        print(
            f"{n_records} {ref} records at {n_gauges} gauge stations for {n_storms} storms."
        )


def stats_gauges(source, fes):
    df = data_gauges(source, fes)
    print_stats_gauges(df)
    return (
        df
        .melt(
            id_vars=["reference", "model", "record_id"],
            value_vars=GAUGE_METRICS,
            var_name="indicator",
            value_name="value",
        )
        .groupby(["reference", "model", "indicator"])
        .apply(
            lambda v: pd.Series({
                "N": v["value"].size,
                "mean": v["value"].mean(),
                "median": v["value"].median(),
                "17": v["value"].quantile(q=0.17),
                "83": v["value"].quantile(q=0.83),
            })
        )
        .reset_index()
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

    df = stats_gauges(args.source, args.fes)
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
