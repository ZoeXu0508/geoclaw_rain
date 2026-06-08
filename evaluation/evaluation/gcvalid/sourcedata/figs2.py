
import argparse
import pathlib

import pandas as pd

from gcvalid.sourcedata import data_extents, stats_extents
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def stats_extents_overlaps(fes, prot, thresh, models):
    df_all = pd.concat([
        data_extents(src, fes, prot, thresh)
        for src in ["dfo_gfd", "dfo_rapid", "gfd_rapid"]
    ])
    df_all = df_all[df_all["model"].isin(models)].copy()

    df_gc = df_all[df_all["model"] == "geoclaw"].copy()
    n_storms = df_gc.groupby("ibtracs_id").first().shape[0]
    print(f"For {n_storms} storms, there is an overlap of different products.")

    result = []
    for src in ["dfo_gfd", "dfo_rapid", "gfd_rapid"]:
        df = df_all[df_all["source"] == src]
        result.append(stats_extents(df))

    return result


def main():
    parser = argparse.ArgumentParser(description='Plot and summarize results.')
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

    models = ["fm_gfd", "fm_rapid", "geoclaw", "aq_geoclaw", "aq_codec", "climada"]
    l_df = stats_extents_overlaps(args.fes, args.prot, args.thresh, models)
    for panel, df in zip("abc", l_df):
        path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}{panel}.csv"
        print(f"Writing to {path} ...")
        df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
