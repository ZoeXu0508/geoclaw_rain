
import argparse
import pathlib

from gcvalid.sourcedata import data_extents
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


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

    df = data_extents(args.source, args.fes, args.prot, args.thresh)
    df = df[df["model"] == "geoclaw"].copy()
    df = df[["record_id", "tp", "tn", "fp", "fn", "lon_mean", "lat_mean"]]

    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
