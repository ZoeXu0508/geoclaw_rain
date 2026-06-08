"""
Plot one or several gauge time series in a plot with time on the x-axis and water level on y-axis
"""
import numpy as np
import pandas as pd

import gcvalid.plot.common as u_plot
import gcvalid.util.constants as u_const


class PlotTide:
    def __init__(self, figure):
        self.figure = figure


    def plot(self, data, title=None):
        self.figure.clf()
        ax = self.figure.gca()
        for d, kwargs in data:
            ax.plot(d, **kwargs)
        ax.set_xlabel("time")
        ax.set_ylabel("sea level anomaly from annual mean (mm)")
        ymax_all = max([d[0].max() for d in data])
        ymin_all = min([d[0].min() for d in data])
        ymax_ref = data[-1][0].max()
        ymin_ref = data[-1][0].min()
        ymin = ymin_all - 100
        ymax = ymin + 6000
        if ymax < ymax_all:
            ymin = ymin_ref - 500
            ymax = ymin + 6000
            if ymax < ymax_all:
                ymax = ymax_all + 100
                ymin = ymax - 6000
        if np.isfinite(ymin) and np.isfinite(ymax):
            ax.set_ylim(ymin, ymax)
        if title is not None:
            ax.set_title(title)
        if len(data) > 0:
            ax.legend()
        self.figure.canvas.draw_idle()


    def plot_twin(self, data, ylabel=None, ylim=None, yscale=None):
        _ax = self.figure.gca()
        ax = _ax.twinx()
        for d, kwargs in data:
            ax.plot(d, **kwargs)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)
        if yscale is not None:
            # plot a colored scale next to y-axis
            u_plot.plot_scale_along_axis(ax, "y", yscale, twin=True)
        if len(data) > 0:
            h, l = ax.get_legend_handles_labels()
            handles, labels = _ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
            _ax.legend(handles, labels)
        self.figure.canvas.draw_idle()


    def annotate(self, text, point, yoffset=0):
        ax = self.figure.gca()
        ylim = ax.get_ylim()
        xytext = (-30, 0)
        relpos = (1.0, 0.5)
        if point[1] < ylim[0]:
            text = f"{text}: {point[1] / 1000:.1f}m"
            point = (point[0], ylim[0])
            xytext = (0, 30 + yoffset)
            relpos = (1.0, 0.0)
        if point[1] > ylim[1]:
            text = f"{text}: {point[1] / 1000:.1f}m"
            point = (point[0], ylim[1])
            xytext = (0, -30 - yoffset)
            relpos = (1.0, 0.0)
        ax.annotate(text, point, xytext=xytext,
                    textcoords="offset points", ha="right", va="center",
                    arrowprops=dict(arrowstyle='->', relpos=relpos))
        self.figure.canvas.draw_idle()


def plot_gauge_record_comparison(
    ax,
    stdata=None,
    df=None,
    legend=False,
    legend_ax=None,
    legend_kwargs=None,
    mean_corrected=True,
    wind=True,
    meters=False,
    min_range_mm=3500,
    pad_range_mm=100,
    model_names=None,
    model_colors=None,
    title="auto",
    nticks_max=4,
):
    if df is None:
        df = stdata['referenced']
        df.name = stdata["gsrc"]
        df = df.to_frame()
        for sim_data in stdata["simulated"]:
            df[sim_data["model"]] = sim_data["referenced"]
        df = df.dropna()
        if title == "auto":
            title = f"{stdata['gsrc']}:{stdata['filename']} ({stdata['map_id']})"
    model_names = u_const.GAUGE_MODEL_SHORTNAMES if model_names is None else model_names
    model_colors = u_const.GAUGE_MODEL_COLORS if model_colors is None else model_colors
    h_unit_fact = 1e-3 if meters else 1.0

    gsrc = df.columns[0]
    st_series = df[gsrc]
    st_series.name = "measured"

    for model in df.columns[1:]:
        sim_series = df[model]
        sim_series.name = "simulated"
        model_short = model_names[model]
        joined = pd.DataFrame(sim_series).join(st_series, how="inner")
        color = model_colors[model]
        ax.plot(joined['simulated'] * h_unit_fact, color=color, label=model_short)
        if mean_corrected:
            jsim_mean = joined['simulated'].mean()
            jst_mean = joined['measured'].mean()
            ax.plot(
                (joined['simulated'] - jsim_mean + jst_mean) * h_unit_fact,
                linestyle=":",
                color=color,
                label=f"{model_short}, mean-corrected",
            )

    ax.plot(
        st_series * h_unit_fact,
        color=u_const.GAUGE_MODEL_COLORS['observed'],
        label={"gesla3": "GESLA3", "codec": "CoDEC"}[gsrc],
    )

    ax.set_xlabel("Date and time")
    ax.set_ylabel(f"Sea level anomaly\nfrom annual mean ({'m' if meters else 'mm'})")
    ymin_all = df.values.min()
    ymax_ref = st_series.max()
    ymin = ymin_all - pad_range_mm
    ymax = ymin + min_range_mm
    if ymax_ref > ymax:
        ymax = ymax_ref + pad_range_mm
        ymin = ymax - min_range_mm
    ax.set_ylim(ymin * h_unit_fact, ymax * h_unit_fact)

    ticks = ax.get_xticks()
    ticklabels = ax.get_xticklabels()
    n_ticks = len(ticks)
    ch_ticks = (n_ticks // nticks_max) + 1 if n_ticks > nticks_max else 1
    ax.set_xticks(ticks)
    ax.set_xticklabels([l if i % ch_ticks == 0 else "" for i, l in enumerate(ticklabels)])

    if title is not None:
        ax.set_title(
            f"{stdata['gsrc']}:{stdata['filename']} ({stdata['map_id']})"
            if title == "auto" else title
        )

    if wind:
        axt = ax.twinx()
        wind_series = stdata["wind"]["intensity"][st_series.index[0]:st_series.index[-1]]
        axt.plot(wind_series, color="tab:green", label="wind speed", linestyle="--")
        axt.set_ylabel("wind speed (m/s)")
        axt.set_ylim(0, 100)
        u_plot.plot_scale_along_axis(axt, "y", u_const.SAFFIR_SIMPSON_YSCALE, twin=True)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if wind:
            h, l = axt.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        def_legend_kwargs = dict(
            fontsize=7,
            frameon=False,
            ncol=1 if len(handles) < 4 else 3,
            loc="upper right" if len(handles) < 4 else "upper left",
        )
        legend_kwargs = {} if legend_kwargs is None else legend_kwargs
        legend_kwargs = {**def_legend_kwargs, **legend_kwargs}

        legend_ax = ax if legend_ax is None else legend_ax
        legend_ax.legend(handles, labels, **legend_kwargs)
