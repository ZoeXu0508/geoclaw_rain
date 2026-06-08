
import pathlib

import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from gcvalid.figures import MODEL_COLORS, main_setup_mpl, plot_box
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def plot_stats_gauges(gs, data, include_rel=False):
    fig = plt.gcf()
    outer = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs, width_ratios=[5, 1], wspace=0.05,
    )
    ncols = 6 if include_rel else 4
    inner = mgridspec.GridSpecFromSubplotSpec(
        2, ncols, subplot_spec=outer[0, 0],
        height_ratios=[1, 5], hspace=0,
        wspace=1.0,
    )

    panels = [
        (inner[:, :ncols - 2], "a", "Deviation of maximum sea levels"),
        (inner[:, ncols - 2:], "b", "Surge dynamics"),
    ]
    for _gs, panel, title in panels:
        ax = fig.add_subplot(_gs, frameon=False)
        ax.axis("off")
        ax.set_title(title)
        ax.text(
            -0.15, 1.065, panel,
            va="bottom",
            ha="right",
            fontweight="bold",
            fontsize=10,
            transform=ax.transAxes,
        )

    metrics = {
        "dmax": ("Absolute", "Deviation (m)"),
        "dmax_signed": ("Signed", "Deviation (m)"),
        **(
            {
                "dmaxrel": ("Absolute", "Deviation (%)"),
                "dmaxrel_signed": ("Signed", "Deviation (%)"),
            } if include_rel else {}
        ),
        "pearson": ("Pearson", "(Unitless)"),
        "rmse": ("RMSE", "Deviation (m)"),
    }

    configs = [
        ("gesla3", "geoclaw", "GeoClaw\n(comp. to GESLA3)"),
        ("gesla3", "codec", "GTSM\n(comp. to GESLA3)"),
        ("codec", "geoclaw", "GeoClaw\n(comp. to GTSM)"),
    ]

    patch_kwargs = [
        dict(
            facecolor=MODEL_COLORS[model],
            edgecolor=MODEL_COLORS["codec" if ref == "codec" else model],
            hatch="///" if ref == "codec" else "",
            linewidth=0,
        ) for ref, model, _ in configs
    ]

    data = data.set_index(["reference", "model", "indicator"]).to_xarray()

    for i_metric, (metric, (title, ylabel)) in enumerate(metrics.items()):
        ax = fig.add_subplot(inner[1, i_metric])
        for i_config, c in enumerate(configs):
            d = data.sel(reference=c[0], model=c[1], indicator=metric).to_array().to_series()
            if "rel" in metrics:
                d *= 100
            plot_box(ax, i_config, d["median"], d["17"], d["83"], **patch_kwargs[i_config])
        ax.set_xticks([])
        if metric.endswith("_signed"):
            ax.axhline(0, color="black", linewidth=0.5, zorder=10)
            maxy = max(abs(y) for y in ax.get_ylim())
            ax.set_ylim(-maxy, maxy)
        elif metric == "pearson":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.get_yaxis().set_major_locator(mticker.MaxNLocator(nbins=3, steps=[1, 5, 10]))

    labels = [c[2] for c in configs]
    handles = [mpatches.Patch(**patch_kwargs[i_config]) for i_config, _ in enumerate(configs)]
    ax = fig.add_subplot(outer[0, 1], frameon=False)
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(0, 0.2, 1, 1),
        loc="upper left",
        ncol=1,
        frameon=False,
    )


def main():
    main_setup_mpl()

    figsize = (u_const.PLOT_WIDTH_IN, 1.3)
    fig = plt.figure(figsize=figsize)

    gs = mgridspec.GridSpec(1, 1)[0, 0]
    data = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}.csv")
    plot_stats_gauges(gs, data)

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
