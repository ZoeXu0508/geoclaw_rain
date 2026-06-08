
import pathlib

import cartopy.crs as ccrs
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import pandas as pd

from gcvalid.figures import (
    main_setup_mpl,
    plot_raster,
    plot_raster_legend,
    plot_stats_extents,
    plot_stats_hwms,
)
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def main():
    main_setup_mpl()

    figsize = (u_const.PLOT_WIDTH_IN, 7.3)
    fig = plt.figure(figsize=figsize)

    outer = mgridspec.GridSpec(3, 1, height_ratios=[7.8, 3, 2.5], hspace=0.1)
    inner = mgridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[0, 0], height_ratios=[8.7, 1, 2], hspace=0,
    )

    model = "cama"
    ax = fig.add_subplot(inner[0, 0], projection=ccrs.PlateCarree())
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-{model}.tif"
    plot_raster(ax, path, model)
    ax.text(
        0.01, 0.98, "a",
        va="top",
        ha="left",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        color="white",
    )
    ax = fig.add_subplot(inner[1, 0], frameon=False)
    plot_raster_legend(ax)

    models = ["geoclaw", "geoclaw+cama", "cama"]
    data = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-extents.csv")
    plot_stats_extents(outer[1, 0], data, models, legend=False, panels="bcdef")

    data = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-hwms.csv")
    plot_stats_hwms(outer[2, 0], data, models, panels="gh")

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
