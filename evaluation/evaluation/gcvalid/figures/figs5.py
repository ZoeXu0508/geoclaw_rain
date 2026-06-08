
import pathlib

import matplotlib.gridspec as mgridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gcvalid.figures import MODEL_COLORS, MODEL_NAMES, main_setup_mpl
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def plot_scatter(ax, df, reference, model):
    x = df[f"max_{reference}"].values
    y = df[f"max_{model}"].values
    ax.scatter(x, y, color="black", alpha=0.1, s=3)
    ax.add_line(
        mlines.Line2D([0, 1], [0, 1], color="tab:purple", ls="--", lw=0.5, transform=ax.transAxes)
    )

    ax.set_aspect('equal', adjustable='box')
    ax.spines['bottom'].set_color(MODEL_COLORS[reference])
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_color(MODEL_COLORS[model])
    ax.spines['left'].set_linewidth(3)

def main():
    main_setup_mpl()

    figsize = (u_const.PLOT_WIDTH_IN, 2.4)
    fig = plt.figure(figsize=figsize)

    gs = mgridspec.GridSpec(1, 3, wspace=0.45)
    axs = [fig.add_subplot(gs[0, 0])]
    axs += [
        fig.add_subplot(gs[0, 1], sharex=axs[0], sharey=axs[0]),
        fig.add_subplot(gs[0, 2], sharex=axs[0], sharey=axs[0]),
    ]

    data = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}.csv").set_index("record_id")
    # convert mm to m
    data /= 1000

    configs = [
        ("gesla3", "geoclaw", "GeoClaw compared to GESLA3"),
        ("gesla3", "codec", "GTSM compared to GESLA3"),
        ("codec", "geoclaw", "GeoClaw compared to GTSM"),
    ]
    for i_config, (reference, model, title) in enumerate(configs):
        df = data[~data[f"max_{reference}"].isna() & ~data[f"max_{model}"].isna()].copy()
        ax = axs[i_config]
        panel = "abc"[i_config]

        plot_scatter(ax, df, reference, model)
        ax.set_title(title)
        ax.set_xlabel(f"{MODEL_NAMES[reference]} max. sea level (m)")
        ax.set_ylabel(f"{MODEL_NAMES[model]} max. sea level (m)")
        ax.text(
            -0.19, 1.06, panel,
            va="bottom",
            ha="right",
            fontweight="bold",
            fontsize=10,
            transform=ax.transAxes,
        )

    axpad = 0.2
    axmin = np.nanmin(data.values) - axpad
    axmax = np.nanmax(data.values) + axpad
    ax = axs[0]
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
