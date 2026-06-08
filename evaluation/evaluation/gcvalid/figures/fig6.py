
import pathlib

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as mgridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gcvalid.figures import main_setup_mpl, COLORS_WONG
import gcvalid.plot as u_plot
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]

STATION_NAMES = {
    "packery_channel": "Packery Channel, TX",
    "high_island": "High Island, TX",
    "freshwater_canal_locks": "Freshwater Canal Locks, LA",
}


def extent_from_locations(locations):
    bounds = tuple(locations.min(axis=0).tolist() + locations.max(axis=0).tolist())
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    pad = min(0.1 * width, 0.1 * height)
    bounds = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
    return (bounds[0], bounds[2], bounds[1], bounds[3])


def plot_hwms(ax):
    df = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}a.csv")
    proj_data = ccrs.PlateCarree()

    ax.set_title("Coastal USGS high water marks")
    ax.spines["geo"].set_linewidth(0.5)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.1)
    ax.set_extent(extent_from_locations(df[["lon", "lat"]].values), crs=proj_data)
    hwm_heights = df["hwm_above_gnd_m"].values
    threshs = [
        (0.0, "#CCCCCC"),
        (0.1, COLORS_WONG[4]),
        (1.0, COLORS_WONG[1]),
        (2.0, COLORS_WONG[6]),
    ]
    for i, (th, col) in enumerate(threshs):
        th_mask = hwm_heights >= th
        if i + 1 < len(threshs):
            th_mask &= hwm_heights < threshs[i + 1][0]
        ax.scatter(
            df.loc[th_mask, "lon"].values,
            df.loc[th_mask, "lat"].values,
            transform=proj_data,
            color=col,
            linewidths=0,
            marker="D",
            s=10,
            zorder=i,
        )
    u_plot.ax_add_ticks(ax, proj_data, stepsize_x=1.0, stepsize_y=1.0)
    handles = [mpatches.Patch(facecolor=col) for _, col in threshs]
    labels = [f"> {th:.1f} m" for th, _ in threshs]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
    )


def plot_gauge_stations(ax):
    df = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}b.csv")
    proj_data = ccrs.PlateCarree()

    ax.set_title("Tide gauge locations")
    ax.spines["geo"].set_linewidth(0.5)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.1)
    extent = extent_from_locations(df[["lon", "lat"]].values)
    extent = (extent[0], extent[1], extent[2] - 0.63, extent[3] + 0.6)
    ax.set_extent(extent, crs=proj_data)
    st_heights = df["max_obs"].values / 1000
    threshs = [
        (0.0, "#CCCCCC"),
        (0.5, COLORS_WONG[4]),
        (1.0, COLORS_WONG[1]),
        (1.5, COLORS_WONG[6]),
    ]
    scatter_kwargs = {
        "gesla3": dict(marker=".", s=75, linewidths=0.1),
        "codec": dict(marker="s", s=15, linewidths=0),
    }
    for ref in ["codec", "gesla3"]:
        ref_mask = (df["reference"] == ref).values
        st_heights_ref = st_heights[ref_mask]
        locs_ref = df.loc[ref_mask, ["lon", "lat"]].values
        for i, (th, col) in enumerate(threshs):
            th_mask = st_heights_ref >= th
            if i + 1 < len(threshs):
                th_mask &= st_heights_ref < threshs[i + 1][0]
            ax.scatter(
                locs_ref[th_mask, 0], locs_ref[th_mask, 1],
                transform=proj_data,
                color=col,
                edgecolor="black",
                zorder=i,
                **scatter_kwargs[ref],
            )

    stargs = {
        "packery_channel": ([1, -1], "left"),
        "high_island": ([0.1, -0.8], "center"),
        "freshwater_canal_locks": ([-0.35, -1.6], "center"),
    }
    arrowprops = dict(facecolor='black', width=0.5, headwidth=3, headlength=4, shrink=0.05)
    for stname, (arrow_dir, ha) in stargs.items():
        longname = STATION_NAMES[stname]
        arrow_dir = np.array(arrow_dir)
        _df = df[df["stname"].str.startswith(stname)]
        xy = _df[["lon", "lat"]].values[0, :] + 0.05 * arrow_dir
        xytext = xy + arrow_dir
        ax.annotate(longname, xy, xytext, va="top", ha=ha, transform=proj_data)
        ax.annotate("", xy, xytext, transform=proj_data, arrowprops=arrowprops)

    u_plot.ax_add_ticks(ax, proj_data, stepsize_x=2.0, stepsize_y=2.0, right=True)
    handles = [mpatches.Patch(facecolor=col) for _, col in threshs]
    labels = [f"> {th:.1f} m" for th, _ in threshs]
    h_legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
    )
    line2d_kwargs = {
        "gesla3": dict(marker=".", markersize=10, markeredgewidth=0.1),
        "codec": dict(marker="s", markersize=5, markeredgewidth=0),
    }
    ax.legend(
        [
            mlines.Line2D(
                [0], [0], linewidth=0, color="black", markerfacecolor=threshs[0][1], **kwargs,
            ) for kwargs in line2d_kwargs.values()
        ],
        ["GESLA3", "GTSM"],
        loc="upper right",
        frameon=False,
        ncol=2,
    )
    ax.add_artist(h_legend)


def plot_water_levels(ax, panel, title, legend_ax):
    df = pd.read_csv(
        u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}{panel}.csv",
        parse_dates=["date_time"],
    )
    u_plot.plot_gauge_record_comparison(
        ax,
        df=df.set_index("date_time"),
        legend=panel == "c",
        legend_ax=legend_ax,
        legend_kwargs=dict(ncol=3, loc="center"),
        mean_corrected=False,
        wind=False,
        meters=True,
        min_range_mm=2300,
        pad_range_mm=100,
        model_names={'codec': "GTSM", 'geoclaw': "GeoClaw"},
        model_colors={'codec': COLORS_WONG[1], 'geoclaw': COLORS_WONG[5]},
        title=title,
        nticks_max=3,
    )
    ax.set_ylim(-0.7, 1.6)
    if panel != "c":
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylabel("")



def main():
    main_setup_mpl()

    figsize = (u_const.PLOT_WIDTH_IN, 5.1)
    fig = plt.figure(figsize=figsize)

    outer = mgridspec.GridSpec(
        2, 2,
        height_ratios=[20, 14], hspace=0.25,
        width_ratios=[21.8, 20], wspace=0.04,
    )
    inner = mgridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=outer[1, :],
        height_ratios=[10, 1], hspace=1.0,
        width_ratios=[1, 5, 5, 5], wspace=0.07,
    )

    proj_plot = ccrs.PlateCarree()
    axs = [
        fig.add_subplot(outer[0, 0], projection=proj_plot),
        fig.add_subplot(outer[0, 1], projection=proj_plot),
        fig.add_subplot(inner[0, 1]),
    ]
    axs.extend([
        fig.add_subplot(inner[0, 2], sharey=axs[-1]),
        fig.add_subplot(inner[0, 3], sharey=axs[-1]),
    ])
    legend_ax = fig.add_subplot(inner[1, 1:], frameon=False)
    legend_ax.axis("off")

    plot_hwms(axs[0])
    plot_gauge_stations(axs[1])

    for ax, panel, (stname, longname) in zip(axs[2:5], "cde", STATION_NAMES.items()):
        plot_water_levels(ax, panel, longname, legend_ax)

    for i_ax, ax in enumerate(axs):
        ax.text(
            0.0,
            1.03 if i_ax < 2 else 1.05,
            "abcde"[i_ax],
            va="bottom",
            ha="left",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
        )

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
