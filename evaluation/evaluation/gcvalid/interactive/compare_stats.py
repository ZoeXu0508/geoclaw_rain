"""
Classes for the interactive plot dashboard in `compare_stats.ipynb`
"""
import itertools
import pickle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from climada.util import log_level
import climada.util.coordinates as u_coord
import geopandas as gpd
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.gridspec as mgridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry

from gcvalid.compare.hwm import read_hwms
import gcvalid.plot as u_plot
import gcvalid.util as util
import gcvalid.util.constants as u_const
import gcvalid.util.gauge as u_gauge
import gcvalid.util.io as u_io


GAUGE_COMPARE_REF = 'gesla3'
"""The reference data set for comparison of heights ('gtsm', 'gesla', 'uhslc', 'wsl', 'gesla3')"""

WORLD_REGIONS = {
    'North America': (-120, 27, -60, 60),
    'Central America': (-120, 0, -60, 27),
    'Southern Indian Ocean': (25, -50, 60, 0),
    'Northern Indian Ocean': (25, 0, 100, 50),
    'West Pacific': (100, 0, 180, 50),
    'South Pacific': (100, -50, 180, 0),
}
"""The geographical bounds (lon_min, lat_min, lon_max, lat_max) for the regional analysis"""


class Dashboard():
    def __init__(self):
        self.w_fm_source = widgets.Dropdown(
            description="Flood map source:",
            options=["gfd", "dfo", "rapid"],
            style={
                'description_width': 'initial',
                'width': '25ex',
            },
        )
        self.w_fm_source.observe(lambda change: self.setup(), names='value')

        self.w_zos = widgets.Dropdown(
            description="Reference sea level:",
            options=[
                f"{zosname}-fes_{tides}"
                for zosname in ["aviso", "0", "mercator"]
                for tides in ["max", "mean", "min", "no"]
            ],
            style={
                'description_width': 'initial',
                'width': '25ex',
            },
        )
        self.w_zos.observe(lambda change: self.setup(), names='value')

        self.w_pluvial = widgets.Dropdown(
            description="Pluvial/bathtub data:",
            options=[
                 ("Without", "without"),
                 # ("With pluvial from DEM", "w_dempixels"),
                 # ("Only pluvial from DEM", "o_dempixels"),
                 # ("With pluvial from catchments", "w_catchments"),
                 # ("Only pluvial from catchments", "o_catchments"),
                 ("With CaMa (ISIMIP3a)", "w_isimip3a"),
                 ("Only CaMa (ISIMIP3a)", "o_isimip3a"),
                 ("Bathtub (CLIMADA)", "bt_climada"),
                 # ("Bathtub (CoDEC, diy)", "bt_codec"),
                 # ("Bathtub (GC-fes_no, diy)", "bt_geoclaw-fes_no"),
                 # ("Bathtub (GC-fes_min, diy)", "bt_geoclaw-fes_min"),
                 # ("Bathtub (GC-fes_mean, diy)", "bt_geoclaw-fes_mean"),
                 # ("Bathtub (GC-fes_max, diy)", "bt_geoclaw-fes_max"),
                 ("Bathtub (CoDEC, aq)", "bt_aq_codec"),
                 ("Bathtub (GC-fes_no, aq)", "bt_aq_geoclaw-fes_no"),
                 ("Bathtub (GC-fes_min, aq)", "bt_aq_geoclaw-fes_min"),
                 ("Bathtub (GC-fes_mean, aq)", "bt_aq_geoclaw-fes_mean"),
                 ("Bathtub (GC-fes_max, aq)", "bt_aq_geoclaw-fes_max"),
            ],
            style={
                'description_width': 'initial',
                'width': '25ex',
            },
        )
        self.w_pluvial.observe(lambda change: self.setup(), names='value')

        self.w_plottype = widgets.Dropdown(
            description='Plot type:',
            options=[
                ('World map', "world_map"),
                ('Overlap pies', "overlap_pies"),
                ('Overlap bars', "overlap_bars"),
                ('Overlap boxes', "overlap_boxes"),
                ('Overlap by distance to coast', "overlap_dist2coast"),
                ('High water marks', 'hwm_overview'),
                ('Gauges overlay', "tg_lines"),
                ('Gauges stats', "tg_fit"),
            ],
            style={
                'description_width': 'initial',
                'width': '30ex',
            },
        )
        self.w_plottype.observe(lambda change: self.update_plot(), names='value')

        self.w_output_fig = widgets.Output()
        self.w_output_txt = widgets.Output()
        self.figure = None
        self.axs = None

        self.setup()

        self.widget = widgets.VBox([
            widgets.HBox([self.w_fm_source, self.w_zos, self.w_pluvial]),
            self.w_plottype,
            self.w_output_fig,
            self.w_output_txt,
        ])


    def setup(self):
        self.w_output_txt.clear_output(wait=True)
        with self.w_output_txt:
            self.pl = PlotSetup(
                source=self.w_fm_source.value,
                zos=self.w_zos.value,
                pluvial=self.w_pluvial.value,
                model_thresh=0,
            )

        self.update_plot()


    def update_plot(self):
        if self.figure is not None:
            plt.close(fig=self.figure)
            self.w_output_fig.clear_output(wait=True)

        plottype = self.w_plottype.value
        if plottype == "overlap_pies":
            return self._update_plot_overlap_pies()

        with self.w_output_fig, plt.ioff():
            figsize = self.pl.figsizes[plottype]
            self.figure = plt.figure(figsize=figsize)
            _, self.axs = getattr(self.pl, plottype)(fig=self.figure)
            if plottype != "world_map":
                self.figure.tight_layout()
            self.figure.canvas.header_visible = False
            display(self.figure.canvas)


    def _update_plot_overlap_pies(self):
        w_overlap_pies_fig_1 = widgets.Output()
        w_overlap_pies_fig_2 = widgets.Output()

        with w_overlap_pies_fig_1, plt.ioff():
            self.figure = plt.figure(
                figsize=self.pl.figsizes["overlap_pies"], constrained_layout=True)
            display(self.figure.canvas)
        self.figure.canvas.header_visible = False

        with w_overlap_pies_fig_2, plt.ioff():
            fig_map = plt.figure(figsize=self.pl.figsizes["compare_map"], constrained_layout=True)
            display(fig_map.canvas)
        fig_map.canvas.header_visible = False

        with self.w_output_fig:
            display(widgets.VBox([w_overlap_pies_fig_1, w_overlap_pies_fig_2]))

        def pick_cb(map_id):
            fig_map.clf()
            self.pl.compare_map(map_id, fig=fig_map)
            fig_map.canvas.draw_idle()
        _, self.axs = self.pl.overlap_pies(fig=self.figure, pick_cb=pick_cb)


def _append_total_overlap_stats(df):
    df_total = df.sum(numeric_only=True).to_frame().transpose()
    df_total["map_id"] = "0000000N00000-0"
    for f in ["", "flooded_"]:
        df_total[f'coastal_{f}fm_p'] = (
            df_total['coastal_fm_total'] / df_total[f'coastal_{f}total']
        )
        for n in ['both', 'gc']:
            df_total[f'coastal_{f}{n}_p'] = (
                df_total[f'coastal_{n}_total'] / df_total[f'coastal_{f}total']
            )
            for i in range(11):
                df_total[f'coastal_{f}{n}{i}_p'] = (
                    df_total[f'coastal_{n}{i}_total'] / df_total[f'coastal_{f}total']
                )
    for v in ["fm", "both", "gc"]:
        df_total[f'coastal_{v}_total'] = 0
    df_total["maxwind"] = 0
    df_total["category"] = -1
    df_total["lat_mean"] = np.nan
    df_total["region"] = "World"
    return pd.concat([df, df_total])


class PlotSetup:
    compare_values = {
        'fm': 1,
        'gc': 2,
        'both': 3,
    }

    figsizes = {
        "world_map": (16, 5.7),
        "overlap_pies": (16, 7),
        "overlap_bars": (16, 10),
        "overlap_boxes": (10, 4),
        "overlap_dist2coast": (16, 10),
        "hwm_overview": (16, 10),
        "tg_lines": (15, 3),
        "tg_fit": (10, 6),
        "compare_map": (16, 7),
    }

    def __init__(self, source, zos, pluvial, model_thresh):
        self.source = source
        self.zos = zos
        self.pluvial = pluvial
        self.model_thresh = model_thresh

        self.compare_dir = u_const.COMPARE_DIR / self.source / self.pluvial

        path = u_const.FLOODMAPS_DIR / self.source / "meta.hdf5"
        self.fm_meta = pd.read_hdf(path).sort_values(by="map_id")

        self.gridcell_dists = None
        self.hwm = None
        self.gaugedata = None

        self._init_compare_df()


    def _init_compare_df(self):
        self.compare_df = u_io.read_compare_df(
            self.source, self.pluvial, self.zos, self.model_thresh,
            apply_filters=True, verbose=True,
        )

        self.compare_df = gpd.GeoDataFrame(
            self.compare_df,
            geometry=[
                shapely.geometry.box(
                    r['lon_min'], r['lat_min'], r['lon_max'], r['lat_max']
                ) for idx, r in self.compare_df.iterrows()
            ],
            crs="epsg:4326",
        )

        self.compare_df["region"] = ""
        for region_name, region_bounds in WORLD_REGIONS.items():
            mask = (
                (self.compare_df["lon_mean"] >= region_bounds[0])
                & (self.compare_df["lon_mean"] <= region_bounds[2])
                & (self.compare_df["lat_mean"] >= region_bounds[1])
                & (self.compare_df["lat_mean"] <= region_bounds[3])
            )
            self.compare_df.loc[mask, "region"] = region_name


    def _init_dist2coast(self):
        all_dists = []
        all_vals = []
        base_path = u_const.COMPARE_DIR / self.source / self.pluvial / self.zos
        for path in base_path.glob(f"*-thresh_0.0.tif"):
            with rasterio.open(path, "r") as src:
                data = src.read(1).ravel()
                flooded_mask = (0 < data) & (data < 50)
                xgrid, ygrid = u_coord.raster_to_meshgrid(src.transform, src.width, src.height)
                lons = xgrid.ravel()[flooded_mask]
                lats = ygrid.ravel()[flooded_mask]
                data = data[flooded_mask]
            with log_level(level='ERROR', name_prefix='climada'):
                all_dists.append(u_coord.dist_to_coast_nasa(lats, lons, highres=True, signed=True))
            masks = {
                'fm': (1 <= data) & (data <= 11),
                'gc': (12 <= data) & (data <= 22),
                'both': (23 <= data) & (data <= 33),
            }
            data[:] = 0
            for key, mask in masks.items():
                data[mask] = self.compare_values[key]
            all_vals.append(data)
        self.gridcell_dists = np.concatenate(all_dists)
        self.gridcell_compvals = np.concatenate(all_vals)


    def _init_hwms(self):
        prefixes = ['gc', 'fm', 'cm', 'bt', "cl"]

        fes = self.zos.split("-fes_")[1]
        self.hwm = read_hwms(
            self.source,
            maps=self.compare_df["map_id"].values,
            add_flooded_status=True,
            as_dataframe=True,
        ).rename(columns={"latitude": "lat", "longitude": "lon", **{
            c: f'{p}_height'
            for p, c in zip(prefixes, [
                f"geoclaw-fes_{fes}", "floodmap", "cama", "bt_aq_codec", "bt_climada",
            ])
        }})

        for p in prefixes:
            if p == "fm":
                self.hwm[f'{p}_nonflood'] = (self.hwm[f'{p}_height'] == 0)
                self.hwm[f'{p}_flooded'] = (self.hwm[f'{p}_height'] > 0)
            else:
                self.hwm[f'{p}_nonflood'] = (self.hwm[f'{p}_height'] <= 0)
                self.hwm[f'{p}_flooded'] = (self.hwm[f'{p}_height'] > 0)
            self.hwm[f'{p}_other'] = (
                (~self.hwm[f'{p}_nonflood'])
                & (~self.hwm[f'{p}_flooded'])
            )

        base_mask = np.ones_like(self.hwm['gc_flooded'].values, dtype=bool)
        self.hwm_num = {}
        for do_invert in itertools.product([True, False], repeat=len(prefixes)):
            mask = base_mask.copy()
            for p, inv in zip(prefixes, do_invert):
                mask &= (~self.hwm[f'{p}_flooded']
                         if inv else
                         self.hwm[f'{p}_flooded'])
            key = "".join(p for p, inv in zip(prefixes, do_invert) if not inv)
            if all(do_invert):
                key = "other"
            self.hwm_num[f"{key}_flooded"] = mask.sum()

        for p in prefixes:
            self.hwm_num[f"{p}+_flooded"] = self.hwm[f'{p}_flooded'].sum()

        mask = (np.sum([self.hwm[f'{p}_flooded'] for p in prefixes], axis=0) > 1)
        self.hwm_num['some_flooded'] = mask.sum()


    def _init_tgdata(self):
        self.gaugedata = []
        maps_wo_gauges = []
        maps_with_gauges = []
        for idx, row in self.fm_meta.iterrows():
            map_id = row['map_id']

            if map_id not in self.compare_df.map_id.values:
                continue
            zos = self.compare_df[self.compare_df.map_id == map_id].zos.values[0]

            gaugedata = u_gauge.load_gaugedata(
                self.source, map_id, by_gsrc=False, referenced=True, geoclaw_zos=zos,
                filter_gsrc=[GAUGE_COMPARE_REF],
            )

            gaugedata = [
                stdata for stdata in gaugedata
                if stdata['wind'] is not None
                and stdata['wind']['intensity'].max() >= 17.5
            ]
            if len(gaugedata) == 0:
                maps_wo_gauges.append(map_id)
                continue
            maps_with_gauges.append(map_id)

            for stdata in gaugedata:
                if stdata['geoclaw'] is not None:
                    [stdata['geoclaw']]= stdata['geoclaw']

            self.gaugedata.extend([
                stdata for stdata in gaugedata
                if stdata['geoclaw'] is not None
                and len(stdata['geoclaw']['time']) > 0
                and np.isfinite(stdata['referenced']).sum() > 0
            ])

        print(f"Missing gauges for {len(maps_wo_gauges)} maps "
              f"vs. {len(maps_with_gauges)} maps with gauges")

        for stdata in self.gaugedata:
            # restrict time to GeoClaw records if available (±12 hours)
            t_pad = np.timedelta64(12, 'h')
            t_start = np.amin([t[0] for t in stdata['geoclaw']['time']]) - t_pad
            t_end = np.amax([t[-1] for t in stdata['geoclaw']['time']]) + t_pad
            t_mask = (
                (stdata['referenced'].index >= t_start)
                & (stdata['referenced'].index <= t_end)
            )
            stdata['referenced'] = stdata['referenced'][t_mask]

        gdata_valid = []
        for stdata in self.gaugedata:
            if np.isfinite(stdata['referenced']).sum() == 0:
                continue

            gc_valid = []
            for i, ser_gc in enumerate(stdata['geoclaw']['referenced']):
                topo_h = (
                    stdata['geoclaw']['topo_height'][i] - stdata['geoclaw']['annual_msl']
                ) * 1000
                base_sl = (
                    stdata['geoclaw']['base_sea_level'][i] - stdata['geoclaw']['annual_msl']
                ) * 1000
                if max(base_sl, topo_h) < 0.9 * ser_gc.max():
                    gc_valid.append(i)

            if len(gc_valid) == 0:
                continue

            for key in stdata['geoclaw']:
                if isinstance(stdata['geoclaw'][key], list):
                    stdata['geoclaw'][key] = [stdata['geoclaw'][key][i] for i in gc_valid]

            gdata_valid.append(stdata)
        self.gaugedata = gdata_valid

        self.gaugestats = {
            'Δtmax': [],
            'Δvmax': [],
            'm_strange': [],
            'gc_strange': [],
        }

        # padding around surge maximum which defines the main surge period
        t_pad = np.timedelta64(8, 'h')

        # padding arround the surge maximum which defines the maximum period to consider
        t_lpad = 3 * t_pad

        # assume that the surge maxima within and outside the surge period relative to the
        # minimum (within t_lpad) have at least this ratio
        min_relative_max_ratio = 1.3

        for stdata in self.gaugedata:
            if stdata["gsrc"] == "gtsm":
                continue

            stdata['m_strange'] = False
            stdata['gc_strange'] = False

            ser = stdata['referenced'].copy()
            idxmax = ser.idxmax()
            ser.index -= idxmax

            # if there is no clear maximum, mark as "strange"
            val_max = ser.loc[np.timedelta64(0, 'h')]
            val_min = ser.min()
            val_max_outside = max(ser.loc[:-t_pad].max(), ser.loc[t_pad:].max())
            max_ratio = (val_max - val_min) / (val_max_outside - val_min)
            if min_relative_max_ratio > max_ratio:
                self.gaugestats['m_strange'].append(stdata['map_id'])
                stdata['m_strange'] = True
                continue

            t_begin = max(ser.index[0], -t_lpad)
            t_end = min(ser.index[-1], t_lpad)
            ser = ser.loc[t_begin:t_end]
            val_min = ser.min()
            stdata['plot_ser'] = ser

            stdata['plot_ser_gc'] = []
            stdata['gc_strange'] = []
            for i, ser_gc in enumerate(stdata['geoclaw']['referenced']):
                ser_gc = ser_gc.copy()
                ser_gc.index -= idxmax
                gc_val_min = ser_gc.min()

                if ser_gc.index[0] > -t_pad or ser_gc.index[-1] < t_pad:
                    # at least t_pad (hours) before/after maximum need to be in GC period
                    stdata['gc_strange'].append(True)
                    continue

                t_begin = max(ser_gc.index[0], -t_lpad)
                t_end = min(ser_gc.index[-1], t_lpad)
                ser_gc = ser_gc.loc[t_begin:t_end]
                assert ser_gc.size > 0

                # if there is no clear maximum, mark as "strange"
                gc_val_max = ser_gc.max()
                gc_idxmax = ser_gc.idxmax()
                gc_max_outside = max(
                    ser_gc.loc[:gc_idxmax - t_pad].max(),
                    ser_gc.loc[gc_idxmax + t_pad:].max(),
                )
                gc_max_ratio = (gc_val_max - gc_val_min) / (gc_max_outside - gc_val_min)
                if min_relative_max_ratio > gc_max_ratio:
                    # no "clear" maximum
                    stdata['gc_strange'].append(True)
                    continue

                self.gaugestats['Δtmax'].append(gc_idxmax)
                self.gaugestats['Δvmax'].append((gc_val_max - val_max) / (val_max - val_min))
                stdata['plot_ser_gc'].append(ser_gc)

                stdata['gc_strange'].append(False)

            # if any of the GC time series is useful, mark as okay
            stdata['gc_strange'] = all(stdata['gc_strange'])
            if stdata['gc_strange']:
                self.gaugestats['gc_strange'].append(stdata['map_id'])


    def world_map(self, fig=None):
        hwms = read_hwms(self.source, maps=self.compare_df["map_id"].values)
        hwm_points = np.array([(h['longitude'], h['latitude']) for h in hwms]).reshape(-1, 2)
        hwm_points = gpd.GeoSeries(
            gpd.points_from_xy(hwm_points[:, 0], hwm_points[:, 1]),
        )

        self.compare_df["year"] = self.compare_df["ibtracs_id"].str.slice(0, 4).astype(int)
        year_counts = (
            self.compare_df[["year"]].groupby("year").size()
            .reindex(np.arange(self.compare_df["year"].min(), self.compare_df["year"].max() + 1))
            .fillna(0)
        )

        fm_rects = self.compare_df.geometry

        gauge_points = u_gauge.load_gauge_locations(self.source, GAUGE_COMPARE_REF)
        gauge_points = gpd.GeoSeries(
            gpd.points_from_xy(gauge_points["lon"], gauge_points["lat"]),
            crs="epsg:4326",
        )

        if fig is None:
            fig = plt.figure(figsize=self.figsizes["world_map"])

        axs = u_plot.plot_compare_geodata(
            fig,
            self.source,
            fm_rects,
            gauge_points,
            hwm_points,
            year_counts,
        )

        return fig, axs


    def overlap_pies(self, fig=None, pick_cb=None):
        if fig is None:
            fig = plt.figure(figsize=self.figsizes["overlap_pies"], constrained_layout=True)

        colors = ['tab:red', 'tab:orange', 'tab:blue']
        pie_locations = {
            'North America': (-105, 45),
            'Central America': (-110, 0),
            'Southern Indian Ocean': (35, -30),
            'Northern Indian Ocean': (70, 30),
            'West Pacific': (100, 40),
            'South Pacific': (130, -40),
        }
        extent = (
            min(-120, self.compare_df["lon_mean"].min() - 10),
            max(165, self.compare_df["lon_mean"].max() + 10),
            -50, 60
        )
        maps_by_region = []
        data = []
        for region_name, region_bounds in WORLD_REGIONS.items():
            d = {
                "centroid": pie_locations[region_name],
                "bounds": region_bounds,
                "locations": [],
                "args": [],
                "kwargs": [],
                "weights": [],
            }
            df_region = self.compare_df[
                self.compare_df["region"] == region_name
            ].sort_values(by=["lon_mean"])
            if df_region.shape[0] == 0:
                continue
            for idx, row in df_region.iterrows():
                d["locations"].append((row['lon_mean'], row['lat_mean']))
                sizes = row[
                    [f'coastal_flooded_{v}_p' for v in ["fm", "both", "gc"]]
                ].values * 100
                d["args"].append(sizes)
                d["kwargs"].append(dict(
                    startangle=90 - 3.6 * (0.5 * sizes[1] + sizes[0]),
                    colors=colors,
                    shadow=True,
                ))
                d["weights"].append(row.coastal_flooded_total)
            data.append(d)
            maps_by_region.append(df_region["map_id"].values)

        sizes = np.array([
            self.compare_df["coastal_fm_area"].sum(),
            self.compare_df["coastal_both_area"].sum(),
            self.compare_df["coastal_gc_area"].sum()
        ], dtype=np.float64)
        sizes_total = sizes.sum()
        sizes_p = 100 * sizes / sizes_total
        data_total = {
            "title": "Global agreement",
            "centroid": (10, 25),
            "arg": sizes,
            "kwargs": dict(
                shadow=True,
                startangle=90 - 3.6 * (0.5 * sizes_p[1] + sizes_p[0]),
                colors=colors,
                labels=["Observed", "Both", "Modelled"],
                autopct=lambda p: f"{int(p * sizes_total / 100):,}\n({p:.1f}%)",
                textprops=dict(size=15),
            ),
            "labelcolors": ["black", "black", "white"],
            "labelsizes": [7, 7, 7],
            "weight": sizes_total,
        }

        def _pick_cb(i_region, i_pie):
            return pick_cb(maps_by_region[i_region][i_pie])
        gs = mgridspec.GridSpec(1, 1)
        axs = u_plot.overlap_pies(gs, data, extent, data_total=data_total, pick_cb=_pick_cb)

        return fig, axs


    def compare_map(self, map_id, fig=None):
        if fig is None:
            fig = plt.figure(figsize=self.figsizes["compare_map"], constrained_layout=True)

        row = self.compare_df.set_index("map_id").loc[map_id]
        bounds = (row['lon_min'], row['lat_min'], row['lon_max'], row['lat_max'])
        shape = (round(row['height'] / row["dlat"]), round(row['width'] / row["dlon"]))
        extent = (bounds[0], bounds[2], bounds[1], bounds[3])

        path = (
            u_const.COMPARE_DIR / self.source / self.pluvial
            / self.zos / f"{map_id}-thresh_{self.model_thresh:.1f}.tif"
        )
        with rasterio.Env(VRT_SHARED_SOURCE=False):
            data = u_io.read_raster_with_bounds(
                path, bounds=bounds, resampling=rasterio.warp.Resampling.nearest)
        img = u_plot.compare2rgba(data)

        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(111, projection=proj)
        ax.spines['geo'].set_linewidth(0.5)
        ax.imshow(img, origin='upper', extent=extent)
        ax.coastlines(linewidth=0.5)
        ax.set_extent(extent, crs=proj)
        ax.set_title(map_id)
        u_plot.ax_add_ticks(ax, proj)

        axs = [ax]
        return fig, axs


    def overlap_bars(self, fig=None):
        if fig is None:
            fig = plt.figure(figsize=self.figsizes["overlap_bars"], constrained_layout=True)

        df = self.compare_df.sort_values(by=["region", "maxwind"]).drop(columns=["geometry"])
        df['category'] = util.saffir_simpson_category(df['maxwind'].values)
        df = _append_total_overlap_stats(df)
        mask_no_total = (df["region"].values != "World")

        values = df[
            ['coastal_flooded_fm_p']
            + [f'coastal_flooded_both{i}_p' for i in range(11)]
            + [f'coastal_flooded_gc{i}_p' for i in reversed(range(11))]
        ].values.T * 100
        values_cumsum = np.cumsum(values, axis=0)
        values_cumsum[1:, :] = values_cumsum[:-1, :]
        values_cumsum[0, :] = 0
        values = list(np.stack([values_cumsum, values_cumsum + values], axis=-1))

        values_abs = df[[f'coastal_{v}_total' for v in ["fm", "both", "gc"]]].values.T * 100
        values_cumsum = np.cumsum(values_abs, axis=0)
        values_cumsum[1:, :] = values_cumsum[:-1, :]
        values_cumsum[0, :] = 0
        values_abs = list(np.stack([values_cumsum, values_cumsum + values_abs], axis=-1))

        # choose GC bathtub run according to current zos setting (ignore pluvial)
        fes = self.zos.split("-fes_")[-1]
        bt_mode_gc = f"geoclaw-fes_{fes}"
        bt_df = {
            bt_mode: df[["map_id"]].merge(
                _append_total_overlap_stats(u_io.read_compare_df(
                    self.source, f"bt_aq_{bt_mode}", "aviso-fes_max", self.model_thresh,
                    apply_filters=True, verbose=False,
                )),
                on="map_id",
                how="left",
            )
            for bt_mode in [bt_mode_gc, "codec"]
        }

        axs = u_plot.plot_bars_overview(
            fig,
            [{
                "label": "Colored coastal area (%)",
                "weight": 2,
                "ref_y": None,
                "names": [
                    'Observed only' if i == 0
                    else 'both' if i == 2
                    else 'Model only' if i == 12
                    else None
                    for i, _ in enumerate(values)
                ],
                "values": values,
                "colors": (
                    ['red']
                    + [(1, 0.3 + i * 0.05, 0) for i in reversed(range(11))]
                    + [(i * 0.09, i * 0.09, 1) for i in range(11)]
                ),
                "aside_values": [
                    df[f"coastal_flooded_{s}_p"].values[mask_no_total] * 100
                    for s in ["fm", "both", "gc"]
                ],
                "aside_colors": ['tab:red', 'tab:orange', 'tab:blue'],
           }] + [
               {
                   "label": "CSI (%)",
                   "weight": 2,
                   "values": [
                       _df["coastal_flooded_both_p"].values * 100
                       for _df in [df, bt_df[bt_mode]]
                   ],
                   "names": ["Selected model", f"Bathtub with {bt_mode} forcing"],
                   "colors": ["tab:blue", "tab:red"],
                   "ref_y": 100,
                   "inverted": True,
                    "aside_values": [
                       _df["coastal_flooded_both_p"].values[mask_no_total] * 100
                       for _df in [df, bt_df[bt_mode]]
                    ],
                    "aside_colors": ["tab:blue", "tab:red"],
               }
               for bt_mode in [bt_mode_gc, "codec"]
           ] + [{
                "label": "Colored grid cells",
                "weight": 1,
                "names": ['Observed only', 'both', 'Model only'],
                "values": values_abs,
                "colors": ['tab:red', 'tab:orange', 'tab:blue'],
                "aside_values": [
                    df[f"coastal_{s}_total"].values[mask_no_total] * 100
                    for s in ["fm", "both", "gc"]
                ],
                "aside_colors": ['tab:red', 'tab:orange', 'tab:blue'],
            }, {
                "label": "Latitude",
                "weight": 1,
                "values": [df["lat_mean"].values],
                "aside": False,
            }, {
                "label": "Max. wind\nspeed (m/s)",
                "weight": 1,
                "values": [df["maxwind"].values],
                "colors": [{
                    "values": df['category'].values + 1,
                    "map": u_const.SAFFIR_SIMPSON_COLORS,
                    "names": u_const.SAFFIR_SIMPSON_NAMES_LONG,
                }],
                "aside_values": [df["maxwind"].values[mask_no_total]],
            }],
            x_groups=df['region'].values,
        )

        return fig, axs


    def overlap_boxes(self, fig=None):
        # CSI = critical success index
        if fig is None:
            fig = plt.figure(figsize=self.figsizes["overlap_boxes"])

        colors = {
            "fm": "tab:red",
            "both": "tab:orange",
            "gc": "tab:blue",
        }
        for i, ind in enumerate(colors.keys()):
            values = [self.compare_df[f'coastal_flooded_{ind}_p'].values]
            labels = ["Global"]
            for region_name, region_bounds in WORLD_REGIONS.items():
                x = self.compare_df.loc[
                    self.compare_df["region"] == region_name,
                    f'coastal_flooded_{ind}_p'
                ].values
                values.append(x)
                labels.append(f"{region_name} ({x.size})")
            ax = plt.gca()
            pos = np.arange(1, len(labels) + 1) - 0.25 + i * 0.25
            box = ax.boxplot(values, showmeans=True, positions=pos, widths=0.2)
            for item in ['boxes', 'whiskers', 'fliers', 'caps']:
                for patch in box[item]:
                    patch.set_color(colors[ind])

        ax.set_title("(Dis-)agreement by world region")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        axs = [ax]
        return fig, axs


    def overlap_dist2coast(self, fig=None):
        if self.gridcell_dists is None:
            self._init_dist2coast()

        if fig is None:
            fig = plt.figure(figsize=self.figsizes["overlap_dist2coast"])

        colors = {
            'fm': 'tab:red',
            'gc': 'tab:blue',
            'both': 'tab:orange',
        }

        hist_bins = 1000 * np.arange(
            np.ceil(self.gridcell_dists.min() / 1000),
            np.floor(self.gridcell_dists.max() / 1000) + 0.5,
            0.5,
        )

        gs = mgridspec.GridSpec(3, 1, height_ratios=[3, 2, 2])
        gs_inner = mgridspec.GridSpecFromSubplotSpec(3, 1, gs[0], hspace=0)

        ref_ax = fig.add_subplot(gs_inner[0])
        axs = [
            ref_ax,
            fig.add_subplot(gs_inner[1], sharex=ref_ax),
            fig.add_subplot(gs_inner[2], sharex=ref_ax),
            fig.add_subplot(gs[1], sharex=ref_ax),
            fig.add_subplot(gs[2], sharex=ref_ax),
        ]

        nums = {}
        for ax, (key, value) in zip(axs[:3], self.compare_values.items()):
            dists = self.gridcell_dists[self.gridcell_compvals == value]
            color = colors[key]
            nums[key], _, _ = ax.hist(dists, bins=hist_bins, color=color, alpha=0.5, label=key)

            qs = [0.05, 0.25, 0.500, 0.95, "mean"]
            text_kwargs = dict(fontsize=10, va="top", rotation="vertical")
            text_y = 14000
            for q in qs:
                quantile = np.mean(dists) if q == "mean" else np.quantile(dists, q=q)
                ax.axvline(quantile, linestyle=":", color=color)
                text = "Mean" if q == "mean" else f"Perc. {100 * q:.0f}"
                ax.text(quantile - 200, text_y, text, ha="right", **text_kwargs)
                text = f"{quantile / 1000:.1f} km"
                ax.text(quantile + 200, text_y, text, ha="left", **text_kwargs)

            ax.semilogy()
            ax.set_ylim(5e-1, 2e4)
            ax.set_yticks([1e0, 1e1, 1e2, 1e3])
            ax.grid(zorder=1, axis="y", linestyle=":", color="silver")
            ax.legend()

        plt.setp(axs[0].get_xticklabels(), visible=False)
        plt.setp(axs[1].get_xticklabels(), visible=False)
        axs[2].set_xlabel("Signed distance to coast (m)")
        axs[1].set_ylabel("Number of grid cells")

        ax = axs[3]
        yticks = np.linspace(0, 1.0, 11)
        all_values = np.zeros_like(nums['fm'])
        all_sizes = np.sum([n for n in nums.values()], axis=0)
        for key in ["both", "fm", "gc"]:
            prev = all_values.copy()
            all_values += nums[key] / np.fmax(1, all_sizes)
            ax.fill_between(
                hist_bins[:-1], prev, all_values, facecolor=colors[key],
                label=key, alpha=0.8, zorder=2)
            ax.legend(loc="upper left")
        ax.set_yticks(yticks)
        ax.grid(zorder=1)
        ax.set_xlabel("Exact signed distance to coast (m)")
        ax.set_ylabel("Share of colored grid cells")

        ax = axs[4]
        cumnums = {key: np.cumsum(n[::-1])[::-1] for key, n in nums.items()}
        all_values = np.zeros_like(nums['fm'])
        all_sizes = np.sum([n for n in cumnums.values()], axis=0)
        for key in ["both", "fm", "gc"]:
            prev = all_values.copy()
            all_values += cumnums[key] / all_sizes
            ax.fill_between(
                hist_bins[:-1], prev, all_values, facecolor=colors[key],
                label=key, alpha=0.8, zorder=2)
            ax.legend(loc="upper left")
        ax.set_yticks(yticks)
        ax.grid(zorder=1)
        ax.set_xlabel("Minimum signed distance to coast (m)")
        ax.set_ylabel("Share of colored grid cells")

        return fig, axs


    def hwm_overview(self, fig=None):
        if self.hwm is None:
            self._init_hwms()

        prefixes = ['fm', 'cm', 'bt', 'cl', 'gc']
        colors = {
            'fm': 'tab:red',
            'cm': 'tab:olive',
            'bt': 'tab:orange',
            'cl': 'tab:purple',
            'gc': 'tab:blue',
        }
        names = {
            "fm": "Satellite",
            "cm": "CaMa",
            "bt": "Bathtub",
            "cl": "CLIMADA",
            "gc": "GeoClaw",
        }

        model = (
            "cl" if self.pluvial == "bt_climada"
            else "bt" if self.pluvial.startswith("bt_aq")
            else "cm" if self.pluvial == "o_isimip3a"
            else "gc"
        )
        model_name = names[model]

        if fig is None:
            fig = plt.figure(figsize=self.figsizes["hwm_overview"])

        df = self.hwm.sort_values(by=["map_id", "dist2coast"])
        df['category'] = util.saffir_simpson_category(df['max_wind'].values)

        axs = u_plot.plot_bars_overview(
            fig, [{
                "label": "Flood height\nabove ground (m)",
                "values": [df[c].values for c in ['height_above_gnd_m', f'{model}_height']],
                "colors": ["grey", "tab:blue"],
                "inverted": True,
                "weight": 2,
                "aside_type": "difference",
            }, {
                "label": "Flood height\nabove geoid (m)",
                "values": [
                    df['elev_m'].where(df['height_above_gnd_m'].isna()).values,
                    (df['dem'] + df['gc_height']).where(df['gc_height'] > 0).values,
                ],
                "colors": ["grey", "tab:blue"],
                "inverted": True,
                "weight": 2,
                "aside_type": "difference",
            }, {
                "values": [df[f'{p}_flooded'].values for p in prefixes],
                "names": [names[p] for p in prefixes],
                "colors": [colors[p] for p in prefixes],
                "weight": 1,
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
                "label": "ERA5 surface\nrunoff (mm)",
                "values": [df['runoff'].values],
                "colors": ["blue"],
                "weight": 1,
            }, {
                "label": "Distance to\ncoast (km)",
                "values": [-df['dist2coast'].values / 1000],
                "weight": 1,
            }], x_groups=df['map_id'].values,
        )

        return fig, axs


    def tg_lines(self, fig=None):
        if self.gaugedata is None:
            self._init_tgdata()

        if fig is None:
            fig = plt.figure(figsize=self.figsizes["tg_lines"])
        ax = plt.gca()

        lines = []
        lines_gc = []
        for stdata in self.gaugedata:
            if stdata["gsrc"] == "gtsm":
                continue

            if stdata["gc_strange"] or stdata["m_strange"]:
                continue

            line, = ax.plot(
                stdata["plot_ser"].index / np.timedelta64(1, 'h'),
                stdata["plot_ser"].values,
                linewidth=0.2,
                color='black'
            )
            lines.append(line)
            for ser in stdata["plot_ser_gc"]:
                line, = ax.plot(
                    ser.index / np.timedelta64(1, 'h'),
                    ser.values,
                    linewidth=0.2,
                    color='tab:blue',
                )
                lines_gc.append(line)

        ax.legend(
            [lines[0], lines_gc[0]], ['measured', 'geoclaw']
        )

        axs = [ax]
        return fig, axs


    def tg_fit(self, fig=None):
        if self.gaugedata is None:
            self._init_tgdata()

        if fig is None:
            fig = plt.figure(figsize=self.figsizes["tg_fit"])
        fig_width, fig_height = fig.get_size_inches()
        ratio = fig_width / fig_height

        spec = mgridspec.GridSpec(
            nrows=2, ncols=2,
            width_ratios=[1, ratio * 4],
            height_ratios=[1, 4],
        )
        ax00 = fig.add_subplot(spec[0])
        ax11 = fig.add_subplot(spec[3])
        ax01 = fig.add_subplot(spec[1], sharex=ax11)
        ax10 = fig.add_subplot(spec[2], sharey=ax11)

        gc_tmax_h = [t / np.timedelta64(1, 'h') for t in self.gaugestats['Δtmax']]
        ax11.grid()
        ax11.plot(
            gc_tmax_h,
            self.gaugestats['Δvmax'],
            marker='o',
            markersize=3,
            linestyle="none")

        sizes = [
            len(self.gaugestats['Δtmax']),
            len(self.gaugestats['gc_strange']),
            len(self.gaugestats['m_strange']),
        ]
        sizes_total = sum(sizes)
        _, _, autotexts = ax00.pie(
            sizes,
            shadow=True,
            autopct=lambda p: f"{int(p * sizes_total / 100):,}",
            labels=['valid', 'gc too bad', 'no max in obs'],
            textprops=dict(size=7),
        )
        for autotext in autotexts:
            autotext.set_color('white')
        ax00.axis('equal')

        ax10.violinplot([self.gaugestats['Δvmax']], vert=True, showmedians=True)
        ax10.scatter([1], [np.mean(self.gaugestats['Δvmax'])], label="Mean")
        ax10.legend()
        ax01.violinplot([gc_tmax_h], vert=False, showmedians=True)
        ax01.scatter([np.mean(gc_tmax_h)], [1], label="Mean")
        ax01.legend()

        tlim = np.abs(gc_tmax_h).max() + 5
        vlim = np.abs(self.gaugestats['Δvmax']).max() + 0.5
        ax11.set_xlim(-tlim, tlim)
        ax11.set_xlabel("Δt in hours")
        ax11.set_ylim(-vlim, vlim)
        ax10.set_ylabel("relative height deviation")
        ax10.set_xticks([])
        ax01.set_yticks([])

        axs = [[ax00, ax01], [ax10, ax11]]
        return fig, axs
