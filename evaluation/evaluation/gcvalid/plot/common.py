
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


def compare2rgba(data):
    rgba_data = np.zeros(data.shape + (4,))
    # No flooding:
    rgba_data[data == 0] = np.array([1, 1, 1, 1])[None]
    # FM:
    rgba_data[data == 1] = np.array([1, 0, 0, 1])[None]
    for i in range(11):
        # GeoClaw:
        rgba_data[data == 12 + i] = np.array([(10 - i) * 0.06, (10 - i) * 0.06, 1, 1])[None]
        # Both:
        rgba_data[data == 23 + i] = np.array([1, 0.65 + i * 0.01, 0.05 * (10 - i), 1])[None]
    # Non-coastal both
    rgba_data[data == 252] = np.array([0.94, 0.75, 0.75, 1])[None]
    # Non-coastal GeoClaw
    rgba_data[data == 253] = np.array([0.8, 0.8, 0.8, 1])[None]
    # Non-coastal FM
    rgba_data[data == 254] = np.array([0.94, 0.75, 0.75, 1])[None]
    # Non-coastal
    rgba_data[data == 255] = np.array([0.8, 0.8, 0.8, 1])[None]
    return rgba_data


def plot_scale_along_axis(ax, axis, scale_data, twin=False):
    # plot a colored scale next to axis
    other_axis = "x" if axis == "y" else "y"
    vmin, vmax = getattr(ax, f"get_{axis}lim")()
    vsize = vmax - vmin
    omin, omax = getattr(ax, f"get_{other_axis}lim")()
    osize = omax - omin

    bbox = ax.get_window_extent().transformed(ax.get_figure().dpi_scale_trans.inverted())
    oscaling = osize / (bbox.width if axis == "y" else bbox.height)

    fontsize = 8
    scale_width_in = 0.02 * fontsize
    scale_width = scale_width_in * oscaling
    otext_pad = 0.15 * scale_width
    otext = (omax - otext_pad) if twin else (omin + otext_pad)
    orect = (omax - scale_width) if twin else omin
    prev = vmin
    for name, color, smax in scale_data:
        if vmax <= prev or smax <= vmin:
            continue
        smax = min(vmax, smax)
        rect_args = ((prev, orect), smax - prev, scale_width)
        text_args = (prev + 0.007 * vsize, otext, name)
        text_kwargs = dict(ha="left", va="bottom")
        if axis == "y":
            rect_args = (rect_args[0][::-1], rect_args[2], rect_args[1])
            text_args = (text_args[1], text_args[0], text_args[2])
            text_kwargs["rotation"] = "vertical"
            if twin:
                text_kwargs["ha"] = "right"
        elif twin:
            text_kwargs["va"] = "top"
        ax.add_patch(mpatches.Rectangle(*rect_args, facecolor=color))
        ax.text(*text_args, fontsize=fontsize, **text_kwargs)
        prev = smax

def _tick_stepsize(w, step_count):
    # make sure that step sizes "look good" as decimal numbers
    # and that appx. `step_count` steps are made (between 3 to 10)
    step_count = np.clip(step_count, 3, 10)
    w1, w2 = np.divmod(np.log10(w), 1)
    steps = 0.1 * np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 8, 10, 12.5, 15, 20, 25, 30])
    i = np.searchsorted(np.log10((step_count + 1) * steps), w2, side="left")
    return 10**w1 * steps[min(steps.size - 1, i)]


def ax_add_ticks(ax, crs, stepsize_x=None, stepsize_y=None, right=False):
    extent = ax.get_extent()
    width, height = (extent[1] - extent[0], extent[3] - extent[2])
    aspect_ratio = width / height
    fact = _tick_stepsize(width, 5 * aspect_ratio**0.5) if stepsize_x is None else stepsize_x
    ticks = np.arange(np.ceil(extent[0] / fact), np.floor(extent[1] / fact) + 1) * fact
    ax.set_xticks(ticks, crs=crs)
    fact = _tick_stepsize(height, 5 / aspect_ratio**0.5) if stepsize_y is None else stepsize_y
    ticks = np.arange(np.ceil(extent[2] / fact), np.floor(extent[3] / fact) + 1) * fact
    ax.set_yticks(ticks, crs=crs)
    if right:
        ax.yaxis.tick_right()
    for label in ax.yaxis.get_majorticklabels():
        label.set_rotation(-90 if right else 90)
        label.set_rotation_mode("anchor")
        label.set_ha("center")
        label.set_va("center")
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)


class LinearSegmentedNormalize(mcolors.Normalize):
    """Piecewise linear color normalization."""
    def __init__(self, vthresh):
        """Initialize normalization

        Parameters
        ----------
        vthresh : list
            Equally distributed to the interval [0,1].
        """
        self.vthresh = vthresh
        self.values = np.linspace(0, 1, len(self.vthresh))
        mcolors.Normalize.__init__(self, vmin=vthresh[0], vmax=vthresh[-1], clip=False)

    def __call__(self, value, clip=None):
        return np.ma.masked_array(np.interp(value, self.vthresh, self.values))


def colormap_coastal_dem():
    """Return colormap and normalization for coastal areas of DEMs."""
    cmap_terrain = [
        (0, 0, 0),
        (3, 73, 114),
        (52, 126, 255),
        (146, 197, 222),
        (255, 251, 171),
        (165, 230, 162),
        (27, 149, 29),
        (32, 114, 11),
        (117, 84, 0),
    ]
    cmap_terrain = mcolors.LinearSegmentedColormap.from_list(
        "coastal_dem", [tuple(c / 255 for c in rgb) for rgb in cmap_terrain])
    cnorm_coastal_dem = LinearSegmentedNormalize([-8000, -1000, -10, -5, 0, 5, 10, 100, 1000])
    return cmap_terrain, cnorm_coastal_dem


def map_add_gridlines(
    ax, xlocs, ylocs, extent, transform, robinson_correction=False, xalign="", yalign="",
):
    xmin, xmax, ymin, ymax = extent
    width, height = xmax - xmin, ymax - ymin
    color = (0.1, 0.1, 0.1, 0.6)
    ax.gridlines(
        ylocs=ylocs, xlocs=xlocs, crs=transform,
        linewidth=0.5, linestyle=":", color=color, zorder=20,
    )
    for loc in xlocs:
        if loc <= xmin or loc >= xmax:
            continue
        ha = "left" if "left" in xalign or loc > 0 else "right"
        va = "bottom" if "bottom" in xalign else "top"
        xtext = loc + (1 if ha == "left" else -1) * 0.01 * width
        ytext = (ymax if va == "top" else ymin) + (-1 if va == "top" else 1) * 0.012 * height
        ax.text(
            xtext, ytext,
            f"{abs(loc):.0f}°{'' if loc == 0 else 'W' if loc < 0 else 'E'}",
            ha=ha, va=va,
            color=color,
            zorder=20, transform=transform,
        )
    for loc in ylocs:
        if loc <= ymin or loc >= ymax:
            continue
        x_correction = (
            1.0 / (0.35 * (np.cos(np.radians(loc)) - 1.0) + 1.0)
            if robinson_correction else
            1.0
        )
        ha = "left" if "left" in yalign else "right"
        va = "top" if "top" in yalign or loc > 0 else "bottom"
        xtext = (xmin if ha == "left" else xmax) + (1 if ha == "left" else -1) * 0.005 * width
        ytext = loc + (-1 if va == "top" else 1) * 0.012 * height
        ax.text(
            xtext * x_correction, ytext,
            f"{abs(loc):.0f}°{'' if loc == 0 else 'S' if loc < 0 else 'N'}",
            ha=ha, va=va,
            color=color,
            zorder=20, transform=transform,
        )
