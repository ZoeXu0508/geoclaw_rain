
import pathlib

import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import pandas as pd

from gcvalid.figures import main_setup_mpl, plot_stats_hwms, plot_stats_hwms_filtered
from gcvalid.sourcedata import HWMS_FILTERED_PANEL_CONFIGS
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def main():
    models = ['geoclaw', 'aq_geoclaw', 'aq_codec', 'climada']

    main_setup_mpl()
    figsize = (u_const.PLOT_WIDTH_IN, 6.4)
    fig = plt.figure(figsize=figsize)
    gs = mgridspec.GridSpec(2, 1, height_ratios=[3, 9], hspace=0.3)

    data = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}ab.csv")
    plot_stats_hwms(gs[0, 0], data, ['floodmap', 'dem'] + models)

    data = {
        panel: (
            pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}{panel}.csv"),
            panel_config,
        )
        for panel, panel_config in zip("cdef", HWMS_FILTERED_PANEL_CONFIGS)
    }
    plot_stats_hwms_filtered(gs[1, 0], data, models, legend=False)

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
