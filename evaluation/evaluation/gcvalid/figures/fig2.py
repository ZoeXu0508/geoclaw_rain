
import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

from gcvalid.figures import main_setup_mpl
import gcvalid.plot as u_plot
import gcvalid.util as util
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def main():
    main_setup_mpl()

    figsize = (u_const.PLOT_WIDTH_IN, 2.55)
    fig = plt.figure(figsize=figsize)

    df = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-extents.csv")
    fm_rects = gpd.GeoSeries(
        [
            shapely.geometry.box(
                r['lon_min'], r['lat_min'], r['lon_max'], r['lat_max']
            ) for _, r in df.iterrows()
        ],
        crs="epsg:4326",
    )
    map_id_split = df["map_id"].str.split("_", expand=True)
    df["source"], df["map_id"] = map_id_split[0], map_id_split[1]
    df["ibtracs_id"] = df["map_id"].str.split("-", expand=True)[0]
    df["year"] = df["ibtracs_id"].str.slice(0, 4).astype(int)
    df["saffirsimpson"] = util.saffir_simpson_category(df["maxwind"].values)
    df_storms = (
        df[["ibtracs_id", "year", "saffirsimpson"]]
        .groupby("ibtracs_id")
        .agg({"year": "first", "saffirsimpson": "max"})
        .reset_index()
    )
    df_storms["dummy"] = True
    year_counts = (
        df_storms
        .pivot(index=["year", "ibtracs_id"], columns="saffirsimpson", values="dummy")
        .reindex(columns=np.arange(-1, 6))
        .fillna(False)
        .reset_index()
        .drop(columns=["ibtracs_id"])
        .groupby("year")
        .sum()
    )

    df = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-gauges.csv")
    gauge_points = gpd.GeoSeries(
        gpd.points_from_xy(df["lon"], df["lat"]),
        crs="epsg:4326",
    )

    df = pd.read_csv(u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-hwms.csv")
    hwm_points = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]))

    ax_map_world, ax_time, ax_map_usa = u_plot.plot_compare_geodata(
        fig,
        None,
        fm_rects,
        gauge_points,
        hwm_points,
        year_counts,
    )

    label_args = [
        (ax_map_world, 0.005, 0.99, "a"),
        (ax_map_usa, 0.013, 0.98, "b"),
        (ax_time, -0.046, 1.10, "c"),
    ]
    for ax, x, y, text in label_args:
        ax.text(
            x, y, text,
            va="top",
            ha="right" if text == "c" else "left",
            fontweight="bold",
            fontsize=10,
            transform=ax.transAxes,
        )

    outpath = u_const.PLOT_DIR / f"Figure{FIGURE_NO.upper()}.pdf"
    print(f"Writing to {outpath} ...")
    fig.savefig(outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
