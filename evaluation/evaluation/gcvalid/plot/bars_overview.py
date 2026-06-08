"""
Several rows of bar plots with shared x-axis that display and compare different aspects of a
large set of records (one bar for each record). Useful to get an overview of multi-variate data
sets with between 10 and several hundreds of records.
"""
import matplotlib.collections as mcollections
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def _compare_bars(ax, x, arrs, colors, ref_y=None, inverted=False, labels=None):
    if labels is None:
        labels = [None, None]
    finite_mask = np.isfinite(arrs[0]) & np.isfinite(arrs[1])
    less_mask = (arrs[0] > arrs[1]) if inverted else (arrs[0] < arrs[1])
    v_mins, v_maxs = np.fmin(*arrs), np.fmax(*arrs)
    ax.bar(
        x[finite_mask], (v_maxs - v_mins)[finite_mask], bottom=v_mins[finite_mask],
        color=np.array(colors)[1 - less_mask[finite_mask].astype(int)],
        alpha=0.6, width=0.8,
    )
    if ref_y is not None:
        ax.axhline(y=ref_y, linestyle=":", color="silver", linewidth=1)
    for col, vals, label in zip(colors, arrs, labels):
        ax.scatter(x[finite_mask], vals[finite_mask], color=col, s=3, label=label)
    if not all(l is None for l in labels):
        ax.legend(ncol=2, loc="upper left")


def _stacked_bars(ax, x, values, colors, labels, ref_y=None):
    for label, color, vals in zip(labels, colors, values):
        ax.bar(
            x, vals[:, 1] - vals[:, 0], bottom=vals[:, 0],
            color=color, alpha=0.5, width=0.8, label=label,
        )
    if ref_y is not None:
        ax.axhline(y=ref_y, linestyle=":", color="silver", linewidth=1)
    ax.legend(ncol=len(values), loc="upper left")


def _multicolor_bars(ax, x, values, color_spec):
    if isinstance(color_spec, dict):
        uniq_colors = np.unique(color_spec["values"])
        for i_col in uniq_colors:
            col = color_spec["map"][i_col]
            label = color_spec["names"][i_col] if "names" in color_spec else None
            mask = color_spec["values"] == i_col
            ax.bar(x[mask], values[mask], width=0.8, color=col, label=label)
        if "names" in color_spec:
            ymin, ymax = min(0, values.min()), values.max()
            ax.set_ylim(ymin, ymax + 0.4 * (ymax - ymin))
            ax.legend(
                ncol=uniq_colors.size, loc="upper left", fontsize=6, markerscale=0.5,
            )
    else:
        ax.bar(x, values, width=0.8, color=color_spec)


def _bool_indicators(ax, values, colors, labels=None):
    kwargs = dict(radius=0.5, orientation=np.radians(45))
    for i_arr, (arr, col) in enumerate(zip(values, colors)):
        ax.add_collection(mcollections.PatchCollection(
            [mpatches.RegularPolygon((i_val, i_arr), 4, **kwargs)
             for i_val, val in enumerate(arr) if val],
            facecolor=col, edgecolor="none",
        ))
    ax.set_yticks(np.arange(len(values)))
    if labels is not None:
        ax.set_yticklabels(labels)
    ax.set_ylim(-0.5, len(values) - 0.5)


def _set_grp_xticks(axs, x_groups):
    x_grp_change = (x_groups[1:] != x_groups[:-1]).nonzero()[0]
    x_grp_change = np.concatenate([x_grp_change, [x_groups.size - 1]])
    x_grp_ticklabels = x_groups[x_grp_change]
    x_grp_ticks = x_grp_change + 0.5
    for i, ax in enumerate(axs):
        for x in np.concatenate([[-0.5], x_grp_ticks]):
            ax.axvline(x=x, linestyle="-", color="silver", linewidth=1)
        if i == len(axs) - 1:
            ax.set_xticks(x_grp_ticks)
            ax.set_xticklabels(x_grp_ticklabels, fontsize=6, rotation=65, ha='right')
        if i < len(axs) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)


def _plot_main_ax(ax, x, data):
    if data["values"][0].ndim > 1:
        _stacked_bars(
            ax, x, data["values"], data["colors"], data["names"], ref_y=data["ref_y"])
    elif len(data["values"]) == 1:
        _multicolor_bars(ax, x, data["values"][0], data["colors"][0])
    elif data["values"][0].dtype == bool:
        _bool_indicators(ax, data["values"], data["colors"], labels=data.get("names"))
    elif len(data["values"]) == 2:
        _compare_bars(
            ax, x, data["values"], data["colors"],
            ref_y=data["ref_y"], inverted=data["inverted"],
            labels=data.get("names"),
        )
    else:
        raise NotImplementedError("Too many values")

    if data["log"]:
        ax.semilogy()

    if "label" in data:
        ax.set_ylabel(data["label"])


def _plot_aside_ax(ax, data):
    if not data["aside"]:
        return

    if data["aside_values"][0].dtype == bool:
        bar_y = np.arange(len(data["aside_values"]))
        bar_x = [d.sum() / d.size for d in data["aside_values"]]
        ax.barh(bar_y, bar_x, color=data["aside_colors"])
        ax.barh(bar_y, [1 - xi for xi in bar_x], left=bar_x, color="silver")
    elif data["aside_type"] == "difference":
        assert len(data["aside_values"]) == 2, "Too many values"
        arrs = data["aside_values"]
        finite_mask = np.isfinite(arrs[0]) & np.isfinite(arrs[1])
        pos_mask = finite_mask & (arrs[0] > 0) & (arrs[1] > 0)
        ax.boxplot([
            (arrs[1] - arrs[0])[finite_mask],
            (arrs[1] - arrs[0])[pos_mask],
        ])
        ax.axhline(y=0, linestyle=":", color="silver", linewidth=1)
        ax.set_xticklabels(["fin", ">0"])
        ymax = np.abs(ax.get_ylim()).max()
        ax.set_ylim(-ymax, ymax)
    else:
        box = ax.boxplot([
            d[np.isfinite(d)] if d.ndim == 1 else d[:, 1]
            for d in data["aside_values"]
        ])
        colors = ["grey" if isinstance(c, dict) else c for c in data["aside_colors"]]
        for item in ['boxes', 'whiskers', 'fliers', 'caps']:
            c_repeated = colors if item == 'boxes' else np.repeat(colors, 2)
            for c, patch in zip(c_repeated, box[item]):
                patch.set_color(c)
        if data["aside_ref_y"] is not None:
            ax.axhline(y=data["aside_ref_y"], linestyle=":", color="silver", linewidth=1)
        ax.set_xticks([])
    ax.yaxis.tick_right()


def plot_bars_overview(fig, y_data, x_groups=None, spec=None):
    assert len(y_data) > 0, "Need at least one data set for plotting!"

    if x_groups is None:
        x_groups = np.full((y_data[0]["values"][0].shape[0],), "")
    x_data = np.arange(x_groups.size)

    for i, d in enumerate(y_data):
        d_new = {
            "weight": 1,
            "ref_y": 0,
            "inverted": False,
            "log": False,
            "aside": True,
            "aside_type": "separate",
            **d
        }
        if "colors" not in d_new and len(d_new["values"]) == 1:
            d_new["colors"] = ["grey"]
        d_new["aside_values"] = d_new.get("aside_values", d_new["values"])
        d_new["aside_colors"] = d_new.get("aside_colors", d_new["colors"])
        d_new["aside_ref_y"] = d_new.get("aside_ref_y", d_new["ref_y"])
        y_data[i] = d_new

    gs_rows = len(y_data)
    gs_cols = 2 if any(d["aside"] for d in y_data) else 1
    gs_height_ratios = [d["weight"] for d in y_data]
    gs_width_ratios = [15, 1] if gs_cols == 2 else None
    gs_kwargs = dict(height_ratios=gs_height_ratios, width_ratios=gs_width_ratios)
    if spec is not None:
        gs = mgridspec.GridSpecFromSubplotSpec(gs_rows, gs_cols, spec, **gs_kwargs)
    else:
        gs = mgridspec.GridSpec(gs_rows, gs_cols, **gs_kwargs)

    axs = [fig.add_subplot(gs[0,0])]
    axs.extend([fig.add_subplot(gs[i, 0], sharex=axs[0]) for i in range(1, gs_rows)])

    axs_aside = [
        fig.add_subplot(
            gs[i, 1],
            sharey=axs[i] if d["aside_type"] == "separate" else None,
        ) if d["aside"] else None
        for i, d in enumerate(y_data)
    ]

    for ax, ax_aside, data in zip(axs, axs_aside, y_data):
        _plot_main_ax(ax, x_data, data)
        _plot_aside_ax(ax_aside, data)
    _set_grp_xticks(axs, x_groups)

    return axs
