
import argparse
import pathlib

import numpy as np
import pandas as pd

from gcvalid.sourcedata import data_extents, stats_extents
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def print_stats_extents_regional(df):
    l_regions = ["AP", "PI", "SH"]
    print(f"        # " + " ".join(l_regions))
    df_gc = df[df["model"] == "geoclaw"].copy()
    for per_storm in [True, False]:
        _df = (
            df_gc.groupby(by=["source","ibtracs_id"]).first().reset_index()
            if per_storm else df_gc
        )
        print(f"Number of {'storm' if per_storm else 'map'}s for each satellite-based product:")
        for src in ["dfo", "gfd", "rapid", "all"]:
            if src == "all":
                df_src = (
                    _df.groupby("ibtracs_id").first().reset_index()
                    if per_storm else _df.copy()
                )
            else:
                df_src = _df[_df["source"] == src]
            n_total = df_src.shape[0]
            n_per_region = (
                df_src.groupby("region")["source"].count().reindex(l_regions).fillna(0).values
            )
            print("%5s %3d %2d %2d %2d" % (src, n_total, *n_per_region))

    df_grpby = (
        df_gc.groupby(by=["source", "ibtracs_id"]).first().reset_index().groupby("ibtracs_id")
    )
    msk_overlap = (df_grpby["source"].count() > 1)
    n_total = msk_overlap.sum()
    n_per_region = (
        df_grpby.first()[msk_overlap].groupby("region")["source"].count()
        .reindex(l_regions).fillna(0).values
    )
    print("Storms covered by more than one satellite-based product:")
    print("      %3d %2d %2d %2d" % (n_total, *n_per_region))


def stats_extents_regional(source, fes, prot, thresh, models):
    result = []
    df_all = data_extents(source, fes, prot, thresh)
    df_all = df_all[df_all["model"].isin(models)].copy()
    df_all["region"] = "AP"
    df_all.loc[df_all["lon_mean"] > 0, "region"] = "PI"
    df_all.loc[df_all["lat_mean"] < 0, "region"] = "SH"
    print_stats_extents_regional(df_all)

    for region in ["AP", "PI", "SH"]:
        df = df_all[df_all["region"] == region]
        result.append(stats_extents(df))

    return result


def main():
    parser = argparse.ArgumentParser(description='Plot and summarize results.')
    parser.add_argument(
        '--source', type=str, metavar="SOURCE", default="all",
        choices=['all', 'dfo', 'gfd', 'rapid'],
        help='The flood map source.',
    )
    parser.add_argument(
        '--thresh', type=float, metavar="THRESHOLD", default=0.1,
        help='The threshold for GC flooding to be considered 0.',
    )
    parser.add_argument(
        '--fes', type=str, metavar="FES_SETTING", default="max",
        choices=["no", "min", "mean", "max"], help='The FES2014 setting used in the GC runs.',
    )
    parser.add_argument(
        '--prot', type=str, metavar="PROT", default="flopros",
        choices=["noprot", "2yprot", "flopros"], help='The CaMa protection setting.',
    )
    args = parser.parse_args()

    models = ["geoclaw", "aq_geoclaw", "aq_codec", "climada"]
    l_df = stats_extents_regional(args.source, args.fes, args.prot, args.thresh, models)
    for panel, df in zip("abc", l_df):
        path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}{panel}.csv"
        print(f"Writing to {path} ...")
        df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
