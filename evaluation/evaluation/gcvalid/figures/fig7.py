
import pathlib

import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import pandas as pd

from gcvalid.figures import main_setup_mpl, plot_stats_hwms
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def main():
    main_setup_mpl()

    figsize = (u_const.PLOT_WIDTH_IN, 1.3)
    fig = plt.figure(figsize=figsize)

    gs = mgridspec.GridSpec(1, 1)[0, 0]
    data = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}.csv")
    models = ['floodmap', 'dem', 'geoclaw', 'aq_geoclaw', 'aq_codec', 'climada']
    plot_stats_hwms(gs, data, models)

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
