
import pathlib

import cartopy.crs as ccrs
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt

from gcvalid.figures import (
    main_setup_mpl,
    plot_raster,
    plot_raster_legend,
)
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def main():
    models = ["rapid", "dfo", "geoclaw", "climada", "aq_geoclaw", "aq_codec"]

    main_setup_mpl()

    figsize = (u_const.PLOT_WIDTH_IN, 4.9)
    fig = plt.figure(figsize=figsize)

    outer = mgridspec.GridSpec(4, 2, height_ratios=[10, 10, 10, 1], hspace=0.01, wspace=0.05)

    proj_plot = ccrs.PlateCarree()
    axs = [fig.add_subplot(outer[0], projection=proj_plot)]
    axs += [
        fig.add_subplot(outer[i], projection=proj_plot, sharex=axs[0], sharey=axs[0])
        for i in range(1, len(models))
    ]

    for i_ax, (ax, model) in enumerate(zip(axs, models)):
        path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-{model}.tif"
        plot_raster(ax, path, model)
        ax.text(
            0.01, 0.98, "abcdefg"[i_ax],
            va="top",
            ha="left",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
            color="white",
        )

    for i, ax in enumerate(axs):
        if i % 2 != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        if i // 2 < 2:
            plt.setp(ax.get_xticklabels(), visible=False)

    ax = fig.add_subplot(outer[3, :], frameon=False)
    plot_raster_legend(ax)

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
