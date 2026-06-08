
import pathlib

import cartopy.crs as ccrs
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gcvalid.figures import main_setup_mpl
import gcvalid.plot as u_plot
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def plot_stats_wmap(gs, df):
    df = df.copy()
    df["weight"] = df[["tn", "fn", "tp", "fp"]].sum(axis=1)

    colors = ['tab:red', 'tab:orange', 'tab:blue']
    world_regions = [
        [(-120, 0, -60, 60), (-56, 37), (31, 26, 25)],
        [(25, 0, 180, 50), (50, 37), (33, 34, 30)],
        [(25, -50, 180, 0), (45, -35), (30, 30, 30)],
    ]
    extent = (
        min(-120, df["lon_mean"].min() - 10),
        max(165, df["lon_mean"].max() + 10),
        -50, 60
    )

    data = []
    for i_model, model in enumerate(["gfd", "dfo", "rapid"]):
        df_m = df[df["record_id"].str.startswith(model)].copy()
        for bounds, pie_loc, m_widths in world_regions:
            m_offset = pie_loc[0]
            if i_model > 0:
                m_offset += np.cumsum(m_widths)[i_model - 1] + 3 * i_model
            d = {
                "title": model.upper(),
                "title_size": 6,
                "centroid": (m_offset, pie_loc[1]),
                "width": m_widths[i_model],
                "locations": [],
                "args": [],
                "kwargs": [],
                "weights": [],
            }
            df_region = df_m[
                (df_m["lon_mean"] >= bounds[0])
                & (df_m["lon_mean"] <= bounds[2])
                & (df_m["lat_mean"] >= bounds[1])
                & (df_m["lat_mean"] <= bounds[3])
            ].sort_values(by=["weight"], ascending=False)
            if df_region.shape[0] == 0:
                continue
            for idx, row in df_region.iterrows():
                d["locations"].append((row['lon_mean'], row['lat_mean']))
                sizes = row[["fn", "tp", "fp"]].values
                sizes_p = 100 * sizes / sizes.sum()
                d["args"].append(sizes)
                d["kwargs"].append(dict(
                    startangle=90 - 3.6 * (0.5 * sizes_p[1] + sizes_p[0]),
                    colors=colors,
                    shadow=True,
                ))
                d["weights"].append(row["weight"])
            data.append(d)

    sizes = np.array([df[c].sum() for c in ["tn", "fn", "tp", "fp"]], dtype=np.float64)
    sizes_total = sizes.sum()
    sizes_sel_p = 100 * sizes[1:] / sizes[1:].sum()
    data_total = {
        "centroid": (-90, -20),
        "arg": sizes_sel_p,
        "kwargs": dict(
            shadow=True,
            startangle=90 - 3.6 * (0.5 * sizes_sel_p[1] + sizes_sel_p[0]),
            colors=colors,
            autopct=lambda p: f"{p:.1f}%",
            textprops=dict(size=7),
        ),
        "labels": ["False negative", "True positive", "False positive"],
        "labelcolors": ["white", "white", "white"],
        "labelsizes": [7, 7, 7],
        "weight": sizes_total,
    }

    axs = u_plot.overlap_pies(gs, data, extent, data_total=data_total, size_range=(0.1, 50))
    ax = axs[0]
    proj = ccrs.PlateCarree()
    for args in [
        (-15, 43, "North Atlantic and\nEastern North Pacific (AP)"),
        (84, 43, "Western North Pacific and\nNorthern Indian Ocean (PI)"),
        (78, -29, "Southern Hemisphere (SH)"),
    ]:
        ax.text(*args, transform=proj, va="bottom", ha="center")


def main():
    main_setup_mpl()

    figsize = (u_const.PLOT_WIDTH_IN, 3.0)
    fig = plt.figure(figsize=figsize)

    gs = mgridspec.GridSpec(1, 1)[0, 0]
    data = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}.csv")
    plot_stats_wmap(gs, data)

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")

if __name__ == "__main__":
    main()
