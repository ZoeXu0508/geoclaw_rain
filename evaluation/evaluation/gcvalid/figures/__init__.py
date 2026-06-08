
import cartopy.crs as ccrs
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as mgridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import gcvalid.plot.common as u_plot


COLORS_IBM = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

COLORS_WONG = [
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

COLORS_TOL = [
    "#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255",
]

RASTER_COLORS = {
    "wet": (1, COLORS_WONG[6], "Wet (flooded)"),
    "dry": (0, "white", "Dry (not flooded)"),
    "perm": (255, COLORS_WONG[2], "Permanent water body"),
    "high": (10, "#000000", "Elevation above 10 m"),
    "miss": (2, "#CCCCCC", "Missing values (in observations)"),
}

MODEL_COLORS = {
    'floodmap': COLORS_WONG[0],
    'dem': COLORS_WONG[3],
    'geoclaw': COLORS_WONG[5],
    'aq_geoclaw': COLORS_WONG[4],
    'codec': COLORS_WONG[1],
    'aq_codec': COLORS_WONG[1],
    'climada': COLORS_WONG[6],
    'cama': COLORS_WONG[2],
    'geoclaw+cama': COLORS_WONG[2],
    'gesla3': "#000000",
}

MODEL_NAMES = {
    'floodmap': "Observed",
    'rapid': "Observed (RAPID)",
    'dfo': "Observed (DFO)",
    'gesla3': "GESLA3",
    'dem': "Topography",
    'geoclaw': "GeoClaw",
    'aq_geoclaw': "GeoClaw+Aqueduct",
    'codec': "GTSM",
    'aq_codec': "GTSM+Aqueduct",
    'climada': "CLIMADA",
    'cama': "CaMa-Flood",
    'geoclaw+cama': "GeoClaw+CaMa-Flood",
}


def main_setup_mpl(agg=True):
    if agg:
        matplotlib.use("Agg")
    plt.rcParams.update({
        "lines.markeredgewidth": 0,
        "lines.markersize": 6,
        "lines.linewidth": 1.0,
        "axes.labelpad": 5,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "font.size": 7,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "font.family": "sans-serif",

        # unfortunately, there is no way to set this on a per-patch basis:
        # https://github.com/matplotlib/matplotlib/issues/21108
        "hatch.linewidth": 3,
    })


def plot_bar(ax, x, y, low, high, width=0.8, **kwargs):
    ax.bar([x], [y], width=width, **kwargs)
    if low is not None:
        ax.plot([x, x], [low, high], color="black")


def plot_box(ax, x, y, low, high, width=0.8, **kwargs):
    rect_args = (
        (x - 0.5 * width, low),
        width, high - low,
    )
    ax.add_patch(mpatches.Rectangle(*rect_args, zorder=1, **kwargs))
    ax.add_patch(mpatches.Rectangle(*rect_args, facecolor='none', edgecolor="black", zorder=2))
    ax.plot([x - 0.5 * width, x + 0.5 * width], [y, y], color="black", zorder=2)


def patch_kwargs(model):
    return dict(
        facecolor="#fff" if model == "floodmap" else MODEL_COLORS[model],
        edgecolor=(
            "#000" if model == "floodmap" else
            MODEL_COLORS["geoclaw" if model == "geoclaw+cama" else model]
        ),
        hatch="///" if model == "geoclaw+cama" else "",
        linewidth=1 if model == "floodmap" else 0,
    )


def _raster_as_rgba(path):
    with rasterio.open(path, "r") as src:
        bounds = tuple(src.bounds)
        extent = (bounds[0], bounds[2], bounds[1], bounds[3])
        data = src.read(1)

    data_rgba = np.zeros(data.shape + (4,), dtype=np.float64)
    for v, c, _ in RASTER_COLORS.values():
        data_rgba[data == v, :] = mcolors.to_rgba(c)
    return data_rgba, extent


def plot_raster(ax, path, model):
    name = MODEL_NAMES[model]
    data, extent = _raster_as_rgba(path)

    proj_data = ccrs.PlateCarree()
    plot_extent = (extent[0], extent[1], extent[2], 30.9)

    ax.spines["geo"].set_linewidth(0.5)
    ax.text(
        0.06, 0.958, name,
        va="top", ha="left", transform=ax.transAxes,
        color="white",
    )
    ax.imshow(data, origin="upper", extent=extent, interpolation="nearest")
    # ax.coastlines(linewidth=0.5)
    ax.set_extent(plot_extent, crs=proj_data)
    u_plot.ax_add_ticks(ax, proj_data, stepsize_y=1.0)


def plot_raster_legend(ax):
    labels = [l for _, _, l in RASTER_COLORS.values()]
    handles = [
        mpatches.Patch(
            facecolor=c,
            edgecolor="black" if c == "white" else None,
            linewidth=0.1 if c == "white" else None,
        )
        for _, c, _ in RASTER_COLORS.values()
    ]
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(0, 0, 1, 0.1),
        loc="upper center",
        ncol=3,
        frameon=False,
        handlelength=1.5,
    )


def plot_stats_extents(gs, data, models, legend=True, panels="abcde"):
    fig = plt.gcf()
    outer = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs, height_ratios=[5, 1], hspace=0.15,
    )
    inner = mgridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0, 0], wspace=0.8)

    metrics = {
        "mcc": ("MCC", "(Unitless)"),
        "f1": ("F1", "(Unitless)"),
        "f2": ("F2", "(Unitless)"),
        "tnr": ("TNR", "Share (%)"),
        "bias": ("Bias", "(Unitless)"),
    }

    data = data.set_index(["model", "indicator"]).to_xarray()

    panel_label_kwargs = lambda ax: dict(
        va="bottom",
        ha="left",
        fontweight="bold",
        fontsize=10,
        transform=ax.transAxes,
    )

    for i_metric, (metric, (title, ylabel)) in enumerate(metrics.items()):
        ax = fig.add_subplot(inner[0, i_metric])
        for i_model, model in enumerate(models):
            d = data.sel(model=model, indicator=metric).to_array().to_series()
            if metric in ["tnr"]:
                d *= 100
            plot_box(ax, i_model, d["median"], d["17"], d["83"], **patch_kwargs(model))
            ax.scatter([i_model], [d["total"]], color="black", marker="*", s=10, zorder=3)
        ax.set_xticks([])
        if metric == "bias":
            ax.axhline(0, color="black", linewidth=0.5, zorder=10)
            maxy = min(5.8, max(abs(y) for y in ax.get_ylim()))
            ax.set_ylim(-maxy, maxy)
        elif metric in ["tnr"]:
            ax.set_ylim(50, 100)
        elif metric != "mcc":
            ax.set_ylim(0, ax.get_ylim()[1])
        else:
            ylim = ax.get_ylim()
            ax.set_ylim(min(-0.03, ylim[0]), max(ylim[1], 0.23))
        ax.text(0.0, 1.07, panels[i_metric], **panel_label_kwargs(ax))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.get_yaxis().set_major_locator(
            mticker.MaxNLocator(nbins=5, steps=[1, 2, 10] if metric == "bias" else [1, 10])
        )

    if not legend:
        return

    labels = ["Total"] + [MODEL_NAMES[m] for m in models]
    handles = (
        [mlines.Line2D(
            [], [],
            color="none",
            marker='*',
            markersize=7,
            markeredgecolor="none",
            markerfacecolor="black",
        )] + [mpatches.Patch(**patch_kwargs(m)) for m in models]
    )
    ax = fig.add_subplot(outer[1, 0], frameon=False)
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(-0.02, 0, 1.02, 1),
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        columnspacing=1.5,
        handlelength=1.5,
        handletextpad=0.6,
    )


def plot_stats_hwms(gs, data, models, panels="ab"):
    fig = plt.gcf()
    outer = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs, width_ratios=[3, 1], wspace=0.1,
    )
    inner0 = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0, 0], width_ratios=[2, 5], wspace=0.29,
    )
    inner1 = mgridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=inner0[0, 1], height_ratios=[1, 5], hspace=0, wspace=0.5,
    )
    inner = [inner0[0, 0], inner1[1, 0], inner1[1, 1]]

    panels = [
        (inner0[0, 0], panels[0], ""),  # title is set later
        (inner0[0, 1], panels[1], "Elevation / Inundation height"),
    ]
    for i_panel, (_gs, panel, title) in enumerate(panels):
        ax = fig.add_subplot(_gs, frameon=False)
        ax.axis("off")
        ax.set_title(title)
        ax.text(
            0.0 if i_panel == 1 else 0.0,
            1.065,
            panel,
            va="bottom",
            ha="left",
            fontweight="bold",
            fontsize=10,
            transform=ax.transAxes,
        )

    metrics = {
        "model_flooded": ("Hit rate", "Share (%)"),
        "dinund": ("Error", "Deviation (m)"),
        "dinund_signed": ("Bias", "Deviation (m)"),
    }

    data = data.set_index(["model", "indicator"]).to_xarray()

    for i_metric, (metric, (title, ylabel)) in enumerate(metrics.items()):
        ax = fig.add_subplot(inner[i_metric])
        sel_models = [
            model for model in models
            if metric == "model_flooded" and model != "dem"
            or metric != "model_flooded" and model != "floodmap"
        ]
        for i_model, model in enumerate(sel_models):
            d = data.sel(model=model, indicator=metric).to_array().to_series()
            (plot_bar if metric == "model_flooded" else plot_box)(
                ax,
                i_model,
                d["mean"] * 100 if metric == "model_flooded" else d["median"],
                None if metric == "model_flooded" else d["17"],
                None if metric == "model_flooded" else d["83"],
                **patch_kwargs(model),
            )
        ax.set_xticks([])
        if metric.endswith("_signed"):
            maxy = max(abs(y) for y in ax.get_ylim())
            ax.axhline(0, color="black", linewidth=0.5, zorder=10)
            ax.set_ylim(-maxy, maxy)
        else:
            ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    labels = [MODEL_NAMES[m] for m in models]
    handles = [mpatches.Patch(**patch_kwargs(m)) for m in models]
    ax = fig.add_subplot(outer[0, 1], frameon=False)
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(0, 0.0, 1, 1),
        loc="lower left",
        ncol=1,
        frameon=False,
        borderaxespad=0,
        labelspacing=0.6,
        handletextpad=0.7,
        handlelength=1.8,
    )


def plot_stats_hwms_filtered_single(gs, data, models):
    fig = plt.gcf()
    outer = mgridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs, height_ratios=[1, 5], hspace=0, wspace=0.45,
    )
    inner = [outer[1, 0], outer[1, 1]]

    metrics = {
        "dinund": ("Error", "Deviation (m)"),
        "dinund_signed": ("Bias", "Deviation (m)"),
    }

    data = data.set_index(["model", "indicator"]).to_xarray()

    for i_metric, (metric, (title, ylabel)) in enumerate(metrics.items()):
        ax = fig.add_subplot(inner[i_metric])
        for i_model, model in enumerate(models):
            d = data.sel(model=model, indicator=metric).to_array().to_series()
            plot_box(ax, i_model, d["median"], d["17"], d["83"], **patch_kwargs(model))
        ax.set_xticks([])
        if metric.endswith("_signed"):
            maxy = max(abs(y) for y in ax.get_ylim())
            ax.axhline(0, color="black", linewidth=0.5, zorder=10)
            ax.set_ylim(-maxy, maxy)
        else:
            ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.get_yaxis().set_major_locator(mticker.MaxNLocator(integer=True))


def plot_stats_hwms_filtered(gs, panel_args, models, legend=True):
    fig = plt.gcf()
    outer = mgridspec.GridSpecFromSubplotSpec(
        2 if legend else 1, 1,
        subplot_spec=gs,
        height_ratios=[20, 1] if legend else None,
        hspace=0.1,
    )
    inner = mgridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[0, 0], wspace=0.19, hspace=0.5,
    )

    for i, (panel, (data, panel_config)) in enumerate(panel_args.items()):
        plot_stats_hwms_filtered_single(inner[i], data, models)

        h_above_grnd, exclude_zeros, exclude_by_dtopo = panel_config
        title = (
            "Inundation height above $\\bf{" + ("ground" if h_above_grnd else "geoid")
            + "}$,\n$\\bf{" + ("excluding" if exclude_zeros else "including")
            + "}$ locations modeled as dry"
            + ",\n$\\bf{" + ("excluding" if exclude_by_dtopo else "including")
            + "}$ locations where DEM error is large."
        )
        ax = fig.add_subplot(inner[i], frameon=False)
        ax.axis("off")
        y = 1.30
        ax.text(
            -0.08, y, panel,
            va="top", ha="right",
            fontweight="bold",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.text(
            -0.05, y, title,
            va="top", ha="left",
            fontsize=7,
            transform=ax.transAxes,
        )

    if not legend:
        return

    labels = [MODEL_NAMES[m] for m in models]
    handles = [mpatches.Patch(**patch_kwargs(m)) for m in models]
    ax = fig.add_subplot(outer[1, 0], frameon=False)
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(-0.03, 0, 1.03, 1),
        loc="lower center",
        ncol=len(models),
        frameon=False,
        borderaxespad=0,
        columnspacing=0.6,
        handletextpad=0.7,
        handlelength=1.3,
    )
