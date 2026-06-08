"""
World map (ocean layer) with all data used for the evaluation (including flood map areas, HWMs,
and tide gauge locations)
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import shapely.geometry

import gcvalid.plot as u_plot
import gcvalid.util.constants as u_const


def plot_compare_geodata(fig, source, fm_rects, gauge_points, hwm_points, year_counts):
    proj_data = ccrs.PlateCarree()

    global_bounds = fm_rects.total_bounds
    global_bounds = (
        min(-123, global_bounds[0] - 11),
        min(-31, global_bounds[1] - 2),
        max(151, global_bounds[2] + 1),
        max(51, global_bounds[3] + 2),
    )

    args = (proj_data, fm_rects, gauge_points, hwm_points.set_crs("epsg:4326"))

    ax = fig.add_axes(
        (0.005, 0.0, 0.99, 1),
        projection=ccrs.Robinson(central_longitude=30),
    )
    _plot_regional_geodata(ax, global_bounds, *args, legend=source)
    extent = (global_bounds[0], global_bounds[2], global_bounds[1], global_bounds[3])
    gridline_locs = ([-90, 0, 90], [-35, 0, 35])
    u_plot.map_add_gridlines(
        ax, *gridline_locs, extent, proj_data, robinson_correction=True, yalign="left",
    )

    inax_time = inset_axes(
        ax, "100%", "100%",
        bbox_to_anchor=(0.5, -0.36, 0.50, 0.32),
        bbox_transform=ax.transAxes,
        axes_kwargs=dict(facecolor="none"),
    )
    for spine in ['bottom', 'top', 'left', 'right']:
        inax_time.spines[spine].set_linewidth(0.5)
    _plot_eventcounts(inax_time, year_counts)

    inax_hwms = None
    if hwm_points.shape[0] > 0:
        hwm_bounds = hwm_points.buffer(0.5).total_bounds
        inax_hwms = _prepare_zoom_inset(ax, 0.13, -0.615, hwm_bounds, proj_data)
        _plot_regional_geodata(inax_hwms, hwm_bounds, *args, legend=False)
        inax_hwms.spines['geo'].set_edgecolor("grey")
        inax_hwms.spines['geo'].set_linewidth(1)
        inax_hwms.spines["geo"].set_linestyle("--")
        extent = (hwm_bounds[0], hwm_bounds[2], hwm_bounds[1], hwm_bounds[3])
        u_plot.map_add_gridlines(
            inax_hwms, *gridline_locs, extent, proj_data, xalign="bottom left",
        )

    axs = [ax, inax_time, inax_hwms]
    return axs


def _plot_regional_geodata(
    ax, bounds, proj_data, fm_rects, gauge_points, hwm_points, legend=False,
):
    is_zoomed = bounds[2] - bounds[0] < 100
    lon_mid = 0.5 * (bounds[0] + bounds[2])
    extent = tuple(bounds[i] for i in [0, 2, 1, 3])

    ax.spines['geo'].set_linewidth(0)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.1)
    ax.set_extent(extent, crs=proj_data)

    box = shapely.geometry.box(*bounds)
    if bounds[2] > 180:
        fm_rects = fm_rects.to_crs({"proj": "longlat", "lon_wrap": lon_mid})
        gauge_points = gauge_points.to_crs({"proj": "longlat", "lon_wrap": lon_mid})
    fm_rects = fm_rects[fm_rects.intersects(box)].copy()

    # flood maps
    ax.add_geometries(
        [fm_rects.unary_union], crs=proj_data,
        facecolor=(1, 0, 0, 0.2 if is_zoomed else 0.3)
    )

    # tide gauge stations
    plt_gaugepoints = [
        geom.coords[0] for geom in gauge_points[gauge_points.within(box)].values
    ]
    ax.add_collection(mcollections.PatchCollection(
        [mpatches.Circle(pt, radius=0.1 if is_zoomed else 0.3)
         for pt in plt_gaugepoints],
        facecolor="black", edgecolor='none',
        transform=proj_data))

    # high water marks
    if hwm_points.shape[0] > 0:
        if bounds[2] > 180:
            hwm_points = hwm_points.to_crs({"proj": "longlat", "lon_wrap": lon_mid})
        hwm_points.crs = None
        ax.add_geometries(
            [hwm_points.buffer(0.1 if is_zoomed else 0.5).unary_union],
            crs=proj_data, facecolor="tab:orange", edgecolor='tab:orange', alpha=0.7,
        )

    if legend != False:
        _plot_legend(ax, 0, 0.99, legend)


def _plot_legend(ax, x, y, source):
    line_args = ([], [])
    line_kwargs = dict(color="none", marker='o', markersize=5, markeredgecolor="none")
    handles = [
        mpatches.Rectangle((0, 0), 1, 1, edgecolor="none", facecolor="tab:red", alpha=0.5),
        mlines.Line2D(*line_args, **line_kwargs, markerfacecolor="black"),
        mlines.Line2D(*line_args, **line_kwargs, markerfacecolor="tab:orange"),
    ]
    labels = [
        "Flood extent" if source is None else f"{source.upper()} flood map",
        "GESLA3 tide\ngauge station",
        "USGS high\nwater mark",
    ]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(-0.005, -0.201, 0.15, 0.2),
        ncol=1,
        labelcolor="black",
        facecolor="none",
        edgecolor="none",
        borderpad=0,
        handlelength=1,
        labelspacing=1.5,
    )


def _plot_eventcounts(ax, event_counts):
    if event_counts.values.ndim > 1:
        bottom = np.zeros_like(event_counts.values)
        bottom[:, 1:] = np.cumsum(event_counts.values, axis=1)[:, :-1]
        for i_col in np.arange(bottom.shape[1]):
            ax.bar(
                event_counts.index.values,
                event_counts.values[:, i_col],
                bottom=bottom[:, i_col],
                facecolor=u_const.SAFFIR_SIMPSON_COLORS[i_col],
                alpha=0.7,
                zorder=1,
            )
    else:
        ax.bar(
            event_counts.index.values,
            event_counts.values,
            facecolor="tab:red",
            alpha=0.9,
            zorder=1,
        )

    year_totals = (event_counts.sum(axis=1) if event_counts.values.ndim > 1 else event_counts)
    for i in range(1, year_totals.astype(int).max() + 1):
        ax.axhline(i, linestyle=":", color="#555", linewidth=0.05, zorder=0)

    period = (event_counts.index.values.min(), event_counts.index.values.max())
    xticks = np.arange(period[0], period[1] + 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:d}" if v % 5 == 0 else "" for v in xticks])
    ax.set_xlabel("Year", labelpad=1)

    ax.set_ylabel("Storm count", labelpad=2)
    ax.yaxis.get_major_locator().set_params(integer=True)

    ax.tick_params(axis="both", pad=2)

    if event_counts.values.ndim == 1:
        return

    # add a legend for the Saffir-Simpson color mapping
    handles = [
        mpatches.Rectangle(
            (0, 0), 1, 1,
            edgecolor="none",
            facecolor=u_const.SAFFIR_SIMPSON_COLORS[i_col],
            alpha=0.7,
        )
        for i_col in np.arange(event_counts.shape[1])
    ]
    labels = u_const.SAFFIR_SIMPSON_NAMES_LONG
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(-0.03, -0.67, 0.3, 0.3),
        ncol=4,
        edgecolor="none",
        borderpad=0,
        framealpha=1,
        fancybox=False,
        handlelength=1,
        columnspacing=1.5,
    )


def _prepare_zoom_inset(ax, x, y, bounds, proj_data):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width, height = xmax - xmin, ymax - ymin
    box_x = xmin + x * width
    box_y = ymin + y * height
    aspect = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])
    box_h = 0.6 * height
    box_w = aspect * box_h

    ax.add_patch(
        mpatches.Rectangle(
            (bounds[0], bounds[1]),
            bounds[2] - bounds[0],
            bounds[3] - bounds[1],
            transform=proj_data,
            facecolor="none", edgecolor="grey",
            linewidth=1, linestyle="--",
        ),
    )

    inax = inset_axes(
        ax, "100%", "100%",
        borderpad=0,
        bbox_to_anchor=(box_x, box_y, box_w, box_h),
        bbox_transform=ax.transData,
        axes_class=GeoAxes,
        axes_kwargs=dict(projection=proj_data),
    )
    ax.set_zorder(1)
    inax.set_zorder(0)

    pady = 0.005 * (bounds[3] - bounds[1])
    padx = aspect * pady
    points = [
        [(bounds[0], bounds[1]), (bounds[0] + padx, bounds[3] - pady)],
        [(bounds[2], bounds[1]), (bounds[2] - padx, bounds[3] - pady)],
    ]
    for ptA, ptB in points:
        ax.add_artist(mpatches.ConnectionPatch(
            xyA=ax.projection.transform_point(*ptA, proj_data),
            xyB=inax.projection.transform_point(*ptB, proj_data),
            coordsA='data', coordsB='data',
            axesA=ax, axesB=inax, clip_on=False,
            edgecolor="grey", linewidth=1, linestyle="--",
        ))

    return inax
