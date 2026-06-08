
import pathlib

import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import pandas as pd

from gcvalid.figures import main_setup_mpl, plot_stats_hwms_filtered
from gcvalid.sourcedata import HWMS_FILTERED_PANEL_CONFIGS
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def main():
    models = ['geoclaw', 'aq_geoclaw', 'aq_codec', 'climada']
    data = {
        panel: (
            pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}{panel}.csv"),
            panel_config,
        )
        for panel, panel_config in zip("abcd", HWMS_FILTERED_PANEL_CONFIGS)
    }

    main_setup_mpl()
    figsize = (u_const.PLOT_WIDTH_IN, 4.5)
    fig = plt.figure(figsize=figsize)
    gs = mgridspec.GridSpec(1, 1)[0, 0]

    plot_stats_hwms_filtered(gs, data, models)

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
