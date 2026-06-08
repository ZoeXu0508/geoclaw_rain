"""
Classes for the interactive plot dashboard in `compare_gauges.ipynb`
"""
import pickle

from climada.util.constants import ONE_LAT_KM
import climada.util.coordinates as u_coord
import ipywidgets as widgets
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gcvalid.plot as u_plot
import gcvalid.util as util
import gcvalid.util.constants as u_const


pd.options.display.max_columns = 50


DEFAULT_MODEL = 'geoclaw-zos_aviso-fes_max'


class Dashboard:
    figsizes = {
        "plot_counts_by_var": (12, 5),
        "plot_counts_by_range": (12, 5),
        "plot_counts_by_wind": (12, 5),
        "plot_fit_boxes": (15, 3),
        "plot_misfits": (16, 8),
        "plot_overview":  (16, 7),
    }

    metric_names = {
        "pearson": "Pearson correlation",
        "rmse": "RMSE (mm)",
        "dmean": "Mean shift (abs, mm)",
        "dmax": "Max shift (abs, mm)",
        "dmean_signed": "Mean shift (mm)",
        "dmax_signed": "Max shift (mm)",
    }

    dropdown_style = {
        'description_width': 'initial',
        'width': '25ex',
    }

    def __init__(self):
        self.w_fm_source = widgets.Dropdown(
            description="Flood map source:",
            options=["gfd", "dfo", "rapid"],
            style=self.dropdown_style,
        )
        self.w_fm_source.observe(lambda change: self.setup(), names='value')

        self.w_compare_ref = widgets.Dropdown(
            description="Reference gauge data:",
            options=["gesla3", "codec"],
            style=self.dropdown_style,
        )
        self.w_compare_ref.observe(lambda change: self.setup(), names='value')

        self.w_plottype = widgets.Dropdown(
            description='Plot type:',
            options=[
                ('Overview', "interactive_overview"),
                ('Wind speeds', "plot_counts_by_wind"),
                ('Tidal ranges', "plot_counts_by_range"),
                ('Box plots of fitting', "interactive_fit_boxes"),
                ('Individual records', "interactive_records"),
                ('Distance to coast', "plot_misfits"),
            ],
            style=self.dropdown_style,
        )
        self.w_plottype.observe(lambda change: self.update_plot(), names='value')

        self.w_output_main = widgets.Output()
        self.figure = None
        self.subfigure = None

        self.setup()

        self.widget = widgets.VBox([
            widgets.HBox([self.w_fm_source, self.w_compare_ref, self.w_plottype]),
            self.w_output_main,
        ])


    def update_plot(self):
        if self.figure is not None:
            plt.close(fig=self.figure)
            self.figure = None
        self.w_output_main.clear_output(wait=True)

        with self.w_output_main, plt.ioff():
            plottype = self.w_plottype.value

            if plottype in [
                "interactive_overview",
                "interactive_records",
                "interactive_fit_boxes",
            ]:
                getattr(self, plottype)()
                return

            self.figure = plt.figure(figsize=self.figsizes[plottype])
            getattr(self, plottype)(fig=self.figure)
            self.figure.canvas.header_visible = False
            self.figure.tight_layout()
            display(self.figure.canvas)


    @property
    def gauges_dir(self):
        return u_const.COMPARE_DIR / self.w_fm_source.value / "gauges"


    def setup(self):
        fit_df_raw = pd.read_csv(self.gauges_dir / "stats.csv")
        fit_df_all = fit_df_raw[
            (fit_df_raw['gsrc'] == self.w_compare_ref.value)
            & (fit_df_raw['valid'] | (fit_df_raw['model'] == "codec"))
            & (~fit_df_raw['rmse'].isna())
        ].copy()
        fit_df_all["record_id"] = fit_df_all["map_id"] + "_" + fit_df_all["stname"]
        fit_ds = fit_df_all.set_index(["model", "record_id"]).to_xarray()

        self.fit_df = {
            model: (
                fit_ds
                .sel(model=model)
                .to_dataframe()
                .dropna(subset=["map_id"])
                .sort_values(by=["map_id", "stname"])
                .reset_index(drop=True)
            )
            for model in fit_ds["model"].values
        }

        fit_ds_gc = fit_ds.sel(
            model=[f'geoclaw-zos_aviso-fes_{s}' for s in ["min", "mean", "max"]],
        )
        for ind in ["rmse", "pearson"]:
            fit_ds_ind = fit_ds_gc.sel(record_id=np.isfinite(fit_ds_gc[ind]).any(dim="model"))
            argopt_fun = getattr(fit_ds_ind[ind], "argmin" if ind == "rmse" else "argmax")
            self.fit_df[f"geoclaw-zos_aviso-fes_best{ind}"] = (
                fit_ds_ind
                .isel(argopt_fun(dim=["model"], skipna=True))
                .to_dataframe()
                .dropna(subset=["map_id"])
                .reset_index(drop=True)
            )

        self._init_gaugedata()
        self.update_plot()


    def _init_gaugedata(self):
        self.gaugedata = []
        for path in self.gauges_dir.glob("*.pickle"):
            with path.open("rb") as fp:
                self.gaugedata.extend(
                    [stdata for stdata in pickle.load(fp)
                     if stdata['affected'] and stdata['valid']
                     and stdata['gsrc'] == self.w_compare_ref.value]
                )

        for stdata in self.gaugedata:
            for s in stdata['simulated']:
                s['tides'] = (
                    "codec" if s['model'] == "codec" else s['model'].split("_")[-1]
                )

        id_cols = ["map_id", "gsrc", "stname"]
        gaugedata_ids = pd.DataFrame(
            [tuple(stdata[c.replace("stname", "filename")] for c in id_cols)
             for stdata in self.gaugedata],
            columns=id_cols).reset_index()
        for model, df in self.fit_df.items():
            df['i_gaugedata'] = df[id_cols].merge(
                gaugedata_ids, on=id_cols, how="left",
            )['index'].values

        for m, df in self.fit_df.items():
            df['ref_range'] = [
                self.gaugedata[i]['referenced'].max()
                - self.gaugedata[i]['referenced'].min()
                for i in df['i_gaugedata'].values
            ]
            df['max_wind'] = [
                self.gaugedata[i]['wind']['intensity'].max()
                for i in df['i_gaugedata'].values
            ]
            for prefix in ["", "gc_"]:
                for i_coord, coord in enumerate(["lat", "lon"]):
                    df[f'{prefix}{coord}'] = [
                        self.gaugedata[i][f'{prefix}location'][i_coord]
                        for i in df['i_gaugedata'].values
                    ]
            df['dist_gc'] = u_coord.dist_approx(
                df['lat'].values[:, None], df['lon'].values[:, None],
                df['gc_lat'].values[:, None], df['gc_lon'].values[:, None],
                normalize=False, method="geosphere",
            )[:, 0, 0]
            df['category'] = util.saffir_simpson_category(df['max_wind'].values)


    def _init_quantiles(self, column):
        # fit dataframes by quantiles of tidal ranges or similar indicators
        values = self.fit_df[DEFAULT_MODEL][column].values
        qs = [0.0, 0.333, 0.666, 0.95, 1.000]
        self.fit_df_q = []
        for iq in range(len(qs) - 1):
            range_start = values.min() if qs[iq] == 0 else np.quantile(values, q=qs[iq])
            range_end = values.max() if qs[iq + 1] == 1 else np.quantile(values, q=qs[iq + 1])
            df_q_dict = {
                m: df[(df[column] > range_start) & (df[column] <= range_end)]
                for m, df in self.fit_df.items()
            }
            self.fit_df_q.append((qs[iq], range_start, range_end, df_q_dict))


    def plot_counts_by_var(self, var, fig=None):
        if fig is None:
            fig = plt.figure(figsize=self.figsizes["plot_counts_by_var"])
        ax = fig.add_subplot(1, 1, 1)

        values = self.fit_df[DEFAULT_MODEL][var].values
        bins = {
            "max_wind": np.arange(11, 65, 3),
            "ref_range": np.arange(0, 3500, 200),
        }[var]
        threshs = {
            "max_wind": [
                11, u_const.SAFFIR_SIMPSON_MIN_BY_NAME["C1"],
                u_const.SAFFIR_SIMPSON_MIN_BY_NAME["C2"], 65,
            ],
            "ref_range": [0, 1000, 2000, 3500],
        }[var]
        unit = {
            "max_wind": "m/s",
            "ref_range": "mm",
        }[var]

        ax.hist(values, bins=bins, alpha=0.5)

        text_kwargs = dict(fontsize=12, va="top", ha="center", transform=ax.transAxes)
        text_y = 0.95
        xmin, xmax = ax.get_xlim()
        xwidth = xmax - xmin
        for i in range(3):
            text_x = (0.5 * (threshs[i] + threshs[i + 1]) - xmin) / xwidth
            txt = (
                (values < threshs[i + 1]).sum() if i == 0 else
                ((values >= threshs[i]) & (values <= threshs[i + 1])).sum() if i == 1 else
                (values > threshs[i]).sum()
            )
            ax.text(text_x, text_y, txt, **text_kwargs)
            ax.axvline(threshs[i], linestyle=":", color="k")

        ax.set_xlabel("Tidal range (mm)" if var == "ref_range" else "Wind speed (m/s)")
        ax.set_ylabel(f"Number of {self.w_compare_ref.value} records")

        qs = [0.167, 0.333, 0.500, 0.666, 0.833, 0.95]
        text_kwargs = dict(
            fontsize=12, va="bottom", ha="left", rotation="vertical", color="silver",
        )
        ymin, ymax = ax.get_ylim()
        yheight = ymax - ymin
        text_y = 0.06 * yheight
        for q in qs:
            quantile = np.quantile(values, q=q)
            quantile_txt = f"{quantile:.0f}" if var == "ref_range" else f"{quantile:.1f}"
            ax.axvline(quantile, linestyle=":", color="silver")
            ax.text(
                quantile + 0.005 * xwidth, text_y,
                f"Percentile {100 * q:.0f} ({quantile_txt} {unit})",
                **text_kwargs,
            )

        if var == "max_wind":
            u_plot.plot_scale_along_axis(ax, "x", u_const.SAFFIR_SIMPSON_YSCALE)

        return fig, ax


    def plot_counts_by_range(self, fig=None):
        return self.plot_counts_by_var("ref_range", fig=fig)


    def plot_counts_by_wind(self, fig=None):
        return self.plot_counts_by_var("max_wind", fig=fig)


    def plot_overview(self, fig=None):
        models = [w.value for w in self.w_select_overview_models]

        metric = self.w_select_overview_metric.value
        metric_label = {
            "rmse": "RMSE (mm)",
            "pearson": "Pearson correlation",
            "dmax": "Abs. diff. of max.\nsurge height (mm)",
            "dmean": "Abs. diff. of mean\nsurge height (mm)",
        }[metric]

        df = self.fit_df[models[0]].merge(
            self.fit_df[models[1]][["i_gaugedata", metric]],
            on="i_gaugedata", how="inner", suffixes=("_1", "_2"),
        ).dropna(subset=[f"{metric}_{i}" for i in [1, 2]])
        df['ref_min'] = [self.gaugedata[i]['referenced'].min() for i in df["i_gaugedata"].values]

        sortby = self.w_select_overview_sort.value
        if sortby == "map_fit":
            df["map_fit"] = df.groupby("map_id")[f"{metric}_2"].transform(lambda x: x.median())
            df = df.sort_values(by=["map_fit", "max_wind"])
        elif sortby == "map_location":
            df["map_lat"] = df.groupby("map_id")["lat"].transform(lambda x: x.median())
            df = df.sort_values(by=["map_lat", "lat"])
        else:
            df = df.sort_values(by=sortby)

        ranges = {m: [] for m in models}
        for i in df["i_gaugedata"].values:
            for sim_data in self.gaugedata[i]['simulated']:
                if sim_data["model"] not in models:
                    continue
                ranges[sim_data["model"]].append(
                    (sim_data['referenced'].min(), sim_data['referenced'].max())
                )
        for model in ranges:
            ranges[model] = np.array(ranges[model]) - df['ref_min'].values[:, None]
        ranges["observed"] = np.stack([np.zeros(df.shape[0]), df['ref_range'].values], axis=1)
        range_names = ["observed"] + models

        if fig is None:
            fig = plt.figure(figsize=self.figsizes["plot_overview"])

        axs = u_plot.plot_bars_overview(
            fig, [{
                "label": "Range of surge relative to\nobserved minimum (m)",
                "names": range_names,
                "colors": [u_const.GAUGE_MODEL_COLORS[m] for m in range_names],
                "values": [ranges[m] / 1000 for m in range_names],
                "weight": 3,
            }, {
                "label": metric_label,
                "values": [df[f"{metric}_{i}"].values for i in [1, 2]],
                "colors": [u_const.GAUGE_MODEL_COLORS[m] for m in models],
                "ref_y": 1 if metric == "pearson" else 0,
                "inverted": metric == "pearson",
                "weight": 2,
            }, {
                "label": "Max. wind\nspeed (m/s)",
                "values": [df['max_wind'].values],
                "colors": [{
                    "values": df['category'].values + 1,
                    "map": u_const.SAFFIR_SIMPSON_COLORS,
                    "names": u_const.SAFFIR_SIMPSON_NAMES_LONG,
                }],
                "weight": 1,
            }, {
                "label": "Latitude",
                "values": [df['lat'].values],
                "weight": 1,
                "aside": False,
            }, {
                "label": "Distance\nto virtual\nstation (km)",
                "values": [df['dist_gc'].values],
                "weight": 1,
            }], x_groups=df['map_id'].values if sortby.startswith("map_") else None,
        )

        return fig, axs


    def interactive_overview(self, fig=None):
        def update_plot():
            if self.subfigure is not None:
                plt.close(fig=self.subfigure)
            self.w_output_overview_fig.clear_output(wait=True)
            with self.w_output_overview_fig, plt.ioff():
                self.subfigure = plt.figure(figsize=self.figsizes["plot_overview"])
                self.plot_overview(fig=self.subfigure)
                self.subfigure.canvas.header_visible = False
                self.subfigure.tight_layout()
                display(self.subfigure.canvas)

        self.w_select_overview_sort = widgets.Dropdown(
            description="Sort by:",
            options=["map_location", "map_fit", "max_wind", "ref_range"],
            style=self.dropdown_style,
        )
        self.w_select_overview_sort.observe(lambda change: update_plot(), names='value')

        self.w_select_overview_metric = widgets.Dropdown(
            description="Fit metric:",
            options=[
                ("Pearson correlation", "pearson"),
                ("RMSE (mm)", "rmse"),
                ("Surge height (mm)", "dmax"),
                ("Mean (mm)", "dmean"),
            ],
            style=self.dropdown_style,
        )
        self.w_select_overview_metric.observe(lambda change: update_plot(), names='value')

        self.w_select_overview_models = []
        options = [
            m for m in u_const.GAUGE_MODELS
            if "fes_best" not in m
            and (m != "codec" or self.w_compare_ref.value != "codec")
        ]
        for i in range(2):
            w = widgets.Dropdown(
                description=f"Model {i + 1}:",
                options=options,
                style=self.dropdown_style,
                value=options[0] if i == 0 else DEFAULT_MODEL,
            )
            w.observe(lambda change: update_plot(), names='value')
            self.w_select_overview_models.append(w)

        self.w_output_overview_fig = widgets.Output()

        update_plot()

        self.overview_widget = widgets.VBox([
            widgets.HBox([
                self.w_select_overview_sort, self.w_select_overview_metric,
            ] + self.w_select_overview_models),
            self.w_output_overview_fig,
        ])
        display(self.overview_widget)


    def plot_fit_boxes(self, fig=None):
        if fig is None:
            fig = plt.figure(figsize=self.figsizes["plot_fit_boxes"])
        axs = fig.subplots(1, 5, sharex=True, sharey=True)

        metric = self.w_select_boxes_metric.value
        qcolumn = self.w_select_boxes_qcolumn.value
        fact = 1 if qcolumn == "max_wind" else 1000
        unit = "m/s" if qcolumn == "max_wind" else "m"
        gauge_models = [
            m for m in u_const.GAUGE_MODELS
            if (m != "codec" or self.w_compare_ref.value != "codec")
        ]
        gauge_models_short = [
            ms for m, ms in zip(u_const.GAUGE_MODELS, u_const.GAUGE_MODELS_SHORT)
            if (m != "codec" or self.w_compare_ref.value != "codec")
        ]

        rowdata = [("All records", [self.fit_df[m][metric].dropna().values for m in gauge_models])]
        rowdata += [
            (f"Quantile {100 * q:.0f} ({start / fact:.2f} - {end / fact:.2f}{unit})",
             [df_q[m][metric].dropna().values for m in gauge_models])
            for q, start, end, df_q in self.fit_df_q
        ]

        for i_axcol, (ax, (coltitle, coldata)) in enumerate(zip(axs, rowdata)):
            ax.set_title(coltitle, fontsize=8)
            ax.axvline(
                x=1 if metric == "pearson" else 0,
                linestyle=":", color="silver", linewidth=1,
            )
            box = ax.boxplot(
                coldata,
                vert=False,
                labels=gauge_models_short if i_axcol == 0 else ['' for d in coldata],
                positions=range(len(coldata), 0, -1),
            )
            for item in ['boxes', 'whiskers', 'fliers', 'caps']:
                mod_repeated = gauge_models if item == 'boxes' else np.repeat(gauge_models, 2)
                for m, patch in zip(mod_repeated, box[item]):
                    patch.set_color(u_const.GAUGE_MODEL_COLORS[m])

        ax = axs[0]
        if metric == "pearson":
            ax.set_xlim(-1.1, 1.1)
        elif metric.endswith("_signed"):
            xmax = 1.1 * np.abs(ax.get_xlim()).max()
            ax.set_xlim(-xmax, xmax)

        return fig, axs


    def df_fit_boxes(self):
        metric = self.w_select_boxes_metric.value
        qcolumn = self.w_select_boxes_qcolumn.value
        fact = 1 if qcolumn == "max_wind" else 1000
        unit = "m/s" if qcolumn == "max_wind" else "m"
        gauge_models = [
            m for m in u_const.GAUGE_MODELS
            if (m != "codec" or self.w_compare_ref.value != "codec")
        ]
        gauge_models_short = [
            ms for m, ms in zip(u_const.GAUGE_MODELS, u_const.GAUGE_MODELS_SHORT)
            if (m != "codec" or self.w_compare_ref.value != "codec")
        ]

        fit_df_q = [(np.nan, 0, self.fit_df_q[-1][2], self.fit_df)] + self.fit_df_q
        q_stats = []
        q_keys = []
        for q, start, end, df_q in fit_df_q:
            stats = []
            for model, model_short in zip(gauge_models, gauge_models_short):
                ser = df_q[model][metric]
                stats_df = pd.DataFrame({
                    "mean": [ser.mean()],
                    "median": ser.quantile(q=0.5),
                    "q16": ser.quantile(q=0.16),
                    "q83": ser.quantile(q=0.83),
                })
                if metric != "pearson":
                    # convert from mm to m
                    stats_df /= 1000
                stats_df["model"] = model_short
                stats.append(stats_df)
            stats = pd.concat(stats).set_index("model")
            stats.index.name = None
            q_stats.append(stats)
            q_str = 'All' if np.isnan(q) else f"Quantile {100 * q:.0f}"
            q_keys.append(f"{q_str} ({start / fact:.1f} - {end / fact:.1f}{unit})")
        q_stats = pd.concat(q_stats, axis=1, keys=q_keys)
        return q_stats


    def df_bestrmse_args(self):
        df = self.fit_df['geoclaw-zos_aviso-fes_bestrmse']
        df["fes_setting"] = df['model'].str.slice(22).values
        df_counts = df.groupby("fes_setting").size()
        df_counts.name = "count"
        df_counts = pd.concat([
                df_counts.reset_index(),
                pd.DataFrame({"fes_setting": ["total"], "count": df.shape[0]}),
        ]).set_index("fes_setting")
        df_counts.index.name = None
        return df_counts.transpose()


    def interactive_fit_boxes(self):
        def update_plot():
            metric_name = self.metric_names[self.w_select_boxes_metric.value]
            self._init_quantiles(self.w_select_boxes_qcolumn.value)
            if self.subfigure is not None:
                plt.close(fig=self.subfigure)
            self.w_output_boxes_fig.clear_output(wait=True)
            self.w_output_boxes_tab.clear_output(wait=True)
            with self.w_output_boxes_fig, plt.ioff():
                self.subfigure = plt.figure(figsize=self.figsizes["plot_fit_boxes"])
                self.subfigure.set_label(metric_name)
                self.plot_fit_boxes(fig=self.subfigure)
                self.subfigure.tight_layout()
                display(self.subfigure.canvas)
            with self.w_output_boxes_tab:
                display(
                    self.df_fit_boxes().style
                    .set_caption(metric_name)
                    .format('{:.2f}')
                    .set_table_styles([
                        dict(
                            selector=", ".join(
                                [f"td:nth-child({i})" for i in [5, 9, 13, 17, 21]]
                                + [f"tr:nth-child(1) th:nth-child({i})[colspan]"
                                   for i in range(2, 7)]
                                + [f"tr:nth-child(2) th:nth-child({i})"
                                   for i in [5, 9, 13, 17, 21]]
                            ),
                            props=[("border-right", "1px black solid")],
                        )
                    ])
                )
                display(
                    self.df_bestrmse_args()
                    .style.set_caption("RMSE-optimal FES-settings for GeoClaw")
                    .format('{:.0f}')
                    .set_table_styles([
                        dict(selector="caption", props=[("font-size", "0.8em")])
                    ])
                )

        self.w_select_boxes_qcolumn = widgets.Dropdown(
            description="Quantiles according to:",
            options=["max_wind", "ref_range"],
            style=self.dropdown_style,
        )
        self.w_select_boxes_qcolumn.observe(lambda change: update_plot(), names='value')

        self.w_select_boxes_metric = widgets.Dropdown(
            description="Fit metric:",
            options=[(name, metric) for metric, name in self.metric_names.items()],
            style=self.dropdown_style,
        )
        self.w_select_boxes_metric.observe(lambda change: update_plot(), names='value')

        self.w_output_boxes_fig = widgets.Output()
        self.w_output_boxes_tab = widgets.Output()

        update_plot()

        self.fitbox_widget = widgets.VBox([
            widgets.HBox([
                self.w_select_boxes_qcolumn, self.w_select_boxes_metric,
            ]),
            self.w_output_boxes_fig,
            self.w_output_boxes_tab,
        ])
        display(self.fitbox_widget)


    def interactive_records(self):
        def update_plot():
            st_idx = self.w_select_gauge.value
            if self.subfigure is not None:
                plt.close(fig=self.subfigure)
            self.w_output_gauge_fig.clear_output(wait=True)
            with self.w_output_gauge_fig, plt.ioff():
                self.subfigure = plt.figure(figsize=(10, 5.5))
                ax = self.subfigure.gca()
                u_plot.plot_gauge_record_comparison(ax, stdata=self.gaugedata[st_idx], legend=True)
                self.subfigure.canvas.header_visible = False
                self.subfigure.tight_layout()
                display(self.subfigure.canvas)

        def update_select():
            sortby = self.w_select_gauge_sorting.value
            df_sorted = self.fit_df[DEFAULT_MODEL].sort_values(by=sortby)
            self.w_select_gauge.options = [
                (f"{row['gsrc']}:{row['stname']:.25s} ({row['map_id']})", row['i_gaugedata'])
                for idx, row in df_sorted.iterrows()
            ]
            self.w_select_gauge.value = df_sorted['i_gaugedata'].values[0]


        self.w_select_gauge_sorting = widgets.Dropdown(
            description="Sort by:",
            options=[
                ("Floodmap", ["map_id", "lat"]),
                ("Fit quality", "fit_quality"),
            ],
            style=self.dropdown_style,
        )
        self.w_select_gauge_sorting.observe(lambda change: update_select(), names='value')

        self.w_select_gauge = widgets.Select(options=[], rows=25)
        self.w_select_gauge.layout.width = "58ex"
        self.w_select_gauge.observe(lambda change: update_plot(), names='value')

        self.w_output_gauge_fig = widgets.Output()
        self.w_output_gauge_fig.layout.width = "150ex"

        update_select()

        self.gauge_widget = widgets.HBox([
            widgets.VBox([self.w_select_gauge_sorting, self.w_select_gauge]),
            self.w_output_gauge_fig,
        ])
        display(self.gauge_widget)


    def plot_misfits(self, fig=None):
        if fig is None:
            fig = plt.figure(figsize=self.figsizes["plot_misfits"])
        axs = fig.subplots(3, 1, sharex=True)

        ranking = self.fit_df[DEFAULT_MODEL].sort_values(by="fit_quality")['i_gaugedata'].values

        ref_gsrc = self.w_compare_ref.value

        misfits = np.array([self.gaugedata[i]['gc_loc_misfit'] * ONE_LAT_KM for i in ranking])
        dist2coast = np.array([self.gaugedata[i]['dist2coast'] for i in ranking])
        axs[0].plot(misfits, marker='o', linewidth=0)
        axs[0].set_ylabel(f"Distance between actual ({ref_gsrc})\n and virtual (GeoClaw) tide gauge")
        axs[1].plot(dist2coast, marker='o', linewidth=0)
        axs[1].set_ylabel(f"Distance of {ref_gsrc}\ntide gauge to coast")
        axs[2].plot(np.fmax(0, -dist2coast / misfits), marker='o', linewidth=0)
        axs[2].set_ylabel("Ratio dist2coast / misfit")
        axs[2].semilogy()

        return fig, axs
