
import argparse
import pathlib

from gcvalid.sourcedata import (
    data_extents,
    stats_extents,
    print_stats_extents,
)
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def stats_extents_filtered(source, fes, cama_prot, thresh, models):
    result = []
    df_all = data_extents(source, fes, cama_prot, thresh)
    df_all = df_all[df_all["model"].isin(models)].copy()

    df = df_all.copy()
    df["fm_flooded_area"] = df["tp"] + df["fn"]
    df = df[(df["fm_flooded_area"] >= 100) & (df["fm_flooded_area"] <= 1000)].reset_index()
    print_stats_extents(df)
    result.append(stats_extents(df))

    df = data_extents(source, fes, cama_prot, 0.0)
    print_stats_extents(df)
    result.append(stats_extents(df))

    for src in ["dfo", "gfd", "rapid"]:
        df = df_all[df_all["source"] == src]
        print_stats_extents(df)
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
    l_df = stats_extents_filtered(args.source, args.fes, args.prot, args.thresh, models)
    for panel, df in zip("abcde", l_df):
        path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}{panel}.csv"
        print(f"Writing to {path} ...")
        df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
