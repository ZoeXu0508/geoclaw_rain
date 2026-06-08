
import argparse
import pathlib

import numpy as np
import pandas as pd

from gcvalid.sourcedata import (
    HWMS_FILTERED_PANEL_CONFIGS,
    stats_hwms,
    stats_hwms_filtered,
)
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
        '--fes', type=str, metavar="FES_SETTING", default="max",
        choices=["no", "min", "mean", "max"], help='The FES2014 setting used in the GC runs.',
    )
    parser.add_argument(
        '--prot', type=str, metavar="PROT", default="flopros",
        choices=["noprot", "2yprot", "flopros"], help='The CaMa protection setting.',
    )
    args = parser.parse_args()

    models = ["geoclaw", "aq_geoclaw", "aq_codec", "climada"]

    df = stats_hwms(args.source, args.fes, args.prot, ["floodmap", "dem"] + models, riverine=True)
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}ab.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)

    for panel, panel_config in zip("cdef", HWMS_FILTERED_PANEL_CONFIGS):
        df = stats_hwms_filtered(
            args.source, args.fes, args.prot, models, panel_config, riverine=True,
        )
        path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}{panel}.csv"
        print(f"Writing to {path} ...")
        df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
