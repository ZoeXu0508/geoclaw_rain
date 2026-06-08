"""
World map (ocean layer) with pie charts of flood map statistics
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import gcvalid.plot as u_plot


def overlap_pies(gs, data, extent, data_total=None, pick_cb=None, size_range=(1, 30)):
    """Plot world map with pie charts within inset axes

    Parameters
    ----------
    gs : GridSpec
        A new Axes will be initialized that uses the whole area of this GridSpec. For each pie
        chart, an inset Axes is added.
    data : list of dict
        Each dict describes all the pies in one region:
        * "centroid": the (lon, lat)-center of gravity of the pies
        * "locations": the (lon, lat)-locations connected to the pies
        * "args": wedge sizes for each pie
        * "kwargs": additional keyword arguments passed to matplotlib's "pie" function
        * "weights": display sizes of the pies
    extent : tuple (lon_min, lon_max, lat_min, lat_max)
        Crop the plot to these geographical bounds.
    data_total : dict, optional
        If given, a bigger single separate pie is displayed with:
        * "title": title text
        * "centroid": the (lon, lat)-location of the pie
        * "arg": the wedge sizes of the pie
        * "kwargs": additional keyword arguments passed to matplotlib's "pie" function
        * "labelcolors": colors of the wedge labels (optional)
        * "labelsizes": sizes of the wedge labels (optional)
        * "weight": the size relative to the remaining pies
    pick_cb : function, optional
        If any of the pies is clicked on, this function is called with the region's index in
        `data` and the pie's index in `data["locations"]`.
    size_range : pair of float, optional
        Minimum and maximum size of pie charts, in units of the plot width. Default: (0.001, 0.03)

    Returns
    -------
    axs : list of Axes
        The main Axes that contains the world map, followed by all inset Axes that contain one
        pie chart plot each.
    """
    fig = plt.gcf()

    proj_data = ccrs.PlateCarree()
    proj_plot = ccrs.PlateCarree(central_longitude=0.5 * (extent[0] + extent[1]))

    ax = fig.add_subplot(gs, projection=proj_plot)
    ax.spines['geo'].set_linewidth(0.5)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.1)
    ax.set_extent(extent, crs=proj_data)

    gridline_locs = ([-90, 0, 90], [-35, 0, 35])
    u_plot.map_add_gridlines(ax, *gridline_locs, extent, proj_data)

    map_width, map_height = extent[1] - extent[0], extent[3] - extent[2]
    map_aspect_ratio = map_width / map_height
    max_weight = max(max(d["weights"]) for d in data)

    axs = [ax]
    pies_by_region = []
    for i_region, d in enumerate(data):
        pies_by_region.append([])
        clon, clat = d["centroid"]
        d_lat_min = min(loc[1] for loc in d["locations"])
        pie_loc_rel = (
            (clon - extent[0]) / map_width,
            (clat - extent[2]) / map_height
        )

        row_size_max = d["width"] / map_width
        iax_pad_x = 0.0025
        iax_pad_y = iax_pad_x * map_aspect_ratio
        d["iax_size"] = []
        d["iax_rows"] = [[]]
        len_row = 0.0
        for i_iax, w in enumerate(d["weights"]):
            lbd = (w / max_weight)**0.5
            iax_size = 0.001 * ((1 - lbd) * size_range[0] + lbd * size_range[1])
            iax_size = (iax_size, iax_size * map_aspect_ratio)
            d["iax_size"].append(iax_size)
            if len_row + iax_size[0] > row_size_max:
                d["iax_rows"].append([])
                len_row = 0.0
            d["iax_rows"][-1].append(i_iax)
            len_row += iax_size[0] + iax_pad_x
        d["iax_size"] = np.array(d["iax_size"])
        d["iax_row_size"] = np.array([
            (d["iax_size"][r, 0].sum(), d["iax_size"][r, 1].max())
            for r in d["iax_rows"]
        ])

        region_rect_width = row_size_max + 2 * iax_pad_x
        region_rect_height = (
            d["iax_row_size"][:, 1].sum() + (d["iax_row_size"].shape[0] - 1) * iax_pad_y
        ) + 2 * iax_pad_y
        region_rect_pos = (
            pie_loc_rel[0] - iax_pad_x,
            pie_loc_rel[1] - region_rect_height + iax_pad_y,
        )
        ax.add_patch(mpatches.Rectangle(
            region_rect_pos,
            region_rect_width, region_rect_height,
            transform=ax.transAxes,
            linewidth=0.5, facecolor=(0.2, 0.2, 0.2, 0.4),
        ))
        if "title" in d:
            ax.text(
                region_rect_pos[0],
                region_rect_pos[1] + region_rect_height,
                d["title"],
                transform=ax.transAxes,
                size=d.get("title_size", 15),
                va="bottom",
                ha="left",
            )

        for i_row, row_idx in enumerate(d["iax_rows"]):
            for row_pos, i_iax in enumerate(row_idx):
                lon, lat = d["locations"][i_iax]
                ax.scatter(
                    [lon], [lat],
                    c='k',
                    edgecolor='none',
                    alpha=0.4,
                    s=60 * d["weights"][i_iax] / max_weight,
                    transform=proj_data,
                )

                iax_size = tuple(d["iax_size"][i_iax, :])
                iax_pad_x = (row_size_max - d["iax_row_size"][i_row, 0]) / max(1, len(row_idx) - 1)
                iax_offset_x = d["iax_size"][row_idx, 0][:row_pos].sum() + row_pos * iax_pad_x
                iax_offset_y = -(
                    d["iax_row_size"][:i_row, 1].sum()
                    + 0.5 * d["iax_row_size"][i_row, 1]
                    + i_row * iax_pad_y
                )
                iax_x = pie_loc_rel[0] + iax_offset_x
                iax_y = pie_loc_rel[1] + iax_offset_y - 0.5 * iax_size[1]
                iax_rect = mpatches.Rectangle(
                    (iax_x, iax_y), iax_size[0], iax_size[1],
                    transform=ax.transAxes, picker=True,
                    linewidth=0, facecolor=(0, 0, 0, 0),
                )
                ax.add_patch(iax_rect)

                iax = ax.inset_axes([iax_x, iax_y, iax_size[0], iax_size[1]], picker=True)
                iax.pie(d["args"][i_iax], **d["kwargs"][i_iax])
                iax.axis('equal')

                axs.append(iax)
                pies_by_region[i_region].append(iax_rect)

    if pick_cb is not None:
        def _pie_click_cb(event):
            for i_region, rects_region in enumerate(pies_by_region):
                for i_pie, rect in enumerate(rects_region):
                    if event.artist != rect:
                        continue
                    return pick_cb(i_region, i_pie)
        fig.canvas.mpl_connect("pick_event", _pie_click_cb)

    if data_total is None:
        return axs

    lon, lat = data_total["centroid"]
    lbd = (data_total["weight"] / max_weight)**0.5
    iax_size = 0.001 * ((1 - lbd) * size_range[0] + lbd * size_range[1])
    iax_size = (iax_size, iax_size * map_aspect_ratio)
    iax_x = (lon - extent[0]) / (extent[1] - extent[0])
    iax_y = (lat - extent[2]) / (extent[3] - extent[2])
    iax = ax.inset_axes([
        iax_x - 0.5 * iax_size[0],
        iax_y - 0.5 * iax_size[1],
        iax_size[0],
        iax_size[1]
    ])
    _, _, autotexts = iax.pie(data_total["arg"], **data_total["kwargs"])
    for i, autotext in enumerate(autotexts):
        if "labelcolors" in data_total:
            autotext.set_color(data_total["labelcolors"][i])
        if "labelsizes" in data_total:
            autotext.set_size(data_total["labelsizes"][i])
    if "title" in data_total:
        iax.text(
            0.5, -0.05, data_total["title"],
            transform=iax.transAxes,
            weight="bold",
            size=data_total.get("title_size", 20),
            va="top",
            ha="center",
        )
    iax.axis('equal')
    axs.append(iax)

    handles = [
        mlines.Line2D(
            [], [],
            color="none",
            marker='o',
            markersize=5,
            markeredgecolor="none",
            markerfacecolor=(0, 0, 0, 0.5),
        )
    ] + [mpatches.Patch(facecolor=c) for c in data_total["kwargs"]["colors"]]
    labels = ["Location of flood extent"] + data_total["labels"]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.0, -0.1, 1.0, 0.1),
        ncol=len(labels),
        frameon=False,
    )

    return axs
