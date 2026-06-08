
import pathlib

import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import pandas as pd

from gcvalid.figures import main_setup_mpl, plot_stats_extents
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def main():
    models = ['floodmap', 'geoclaw', 'aq_geoclaw', 'aq_codec', 'climada']

    main_setup_mpl()

    panel_configs = {
        "a": ("Overlap between DFO and GFD", "abcde"),
        "b": ("Overlap between DFO and RAPID", "fghij"),
        "c": ("Overlap between GFD and RAPID", "klmno"),
    }
    n_rows = len(panel_configs)

    figsize = (u_const.PLOT_WIDTH_IN, 1.6 * n_rows + 0.2)
    fig = plt.figure(figsize=figsize)

    outer = mgridspec.GridSpec(n_rows, 1, hspace=0.37)

    panel_label_kwargs = lambda ax: dict(
        va="bottom",
        ha="left",
        fontsize=10,
        transform=ax.transAxes,
    )

    for i_row, (panel, (title, panel_labels)) in enumerate(panel_configs.items()):
        gs = outer[i_row, 0]
        data = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}{panel}.csv")
        data.loc[data["model"].str.startswith("fm_"), "model"] = "floodmap"
        plot_stats_extents(gs, data, models, legend=i_row == n_rows - 1, panels=panel_labels)
        N = data["N"].max()
        ax = fig.add_subplot(gs, frameon=False)
        ax.axis("off")
        ax.text(0.0, 1.27, f"{title} (N={N:.0f})", **panel_label_kwargs(ax))

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
