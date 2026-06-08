"""
Classes for the interactive plot dashboard in `overview.ipynb`
"""
import cartopy.crs as ccrs
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gcvalid.plot as u_plot
import gcvalid.util.constants as u_const
import gcvalid.util.gauge as u_gauge
import gcvalid.util.io as u_io


ALL_GAUGE_SOURCES = ['gesla3', 'gesla', 'uhslc', 'wsl', 'gtsm', 'codec']

GAUGE_SOURCE_COLORS = {
    "gesla3": "gray",
    "codec": "maroon",
    "gtsm": "red",
    "gesla": "black",
    "uhslc": "orange",
    "wsl": "purple",
}

GAUGE_SOURCE_MARKERS = {
    "codec": "x",
    "gtsm": "x",
    "gesla3": ".",
    "gesla": ".",
    "uhslc": ".",
    "wsl": ".",
}


class Dashboard():
    def __init__(self):
        self.proj_data = ccrs.PlateCarree()
        self.proj_ax = ccrs.PlateCarree(central_longitude=0)

        self.w_fm_source = widgets.Dropdown(
            description="Flood map source:",
            options=["gfd", "dfo", "rapid"],
            style={
                'description_width': 'initial',
                'width': '25ex',
            },
        )
        self.w_fm_source.observe(lambda change: self.update_datareader(), names='value')

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
        self.w_zos.observe(lambda change: self.update_datareader(), names='value')

        self.w_toggle_compare_areas = widgets.ToggleButton(
            description="Restrict maps to compare areas", value=True)
        self.w_toggle_compare_areas.observe(lambda change: self.update_datareader(), names="value")

        self.w_year_range = widgets.IntRangeSlider(
            value=[2000, 2019], min=2000, max=2019,
            description="Year range",
            continuous_update=False)
        self.w_year_range.layout.width = "8in"
        self.w_year_range.observe(lambda change: self.filter_tracks(), names='value')

        self.w_select_event = widgets.Dropdown(options=[], description='IBTrACS ID:')
        self.w_select_event.layout.width = "50ex"
        self.w_select_event.observe(lambda change: self.update_single_tc(), names='value')

        self.w_select_map = widgets.Select(options=[], rows=24)
        self.w_select_map.layout.width = "36ex"
        self.w_select_map.observe(lambda change: self.select_map(), names='value')

        self.w_select_map_type = {
             "reference": widgets.Dropdown(options=[
                 ("No reference", "none"),
                 ("DFO raw", "dfo_raw"),
                 ("DFO cleaned", "dfo"),
                 ("RAPID", "rapid"),
                 ("GFD", "gfd"),
             ]),
             "model_inun": widgets.Dropdown(options=[
                 "No inundation model",
                 "GC fgmax",
                 "Bathtub (CLIMADA)",
                 # "Bathtub (CoDEC, diy)",
                 # "Bathtub (GC-fes_no, diy)",
                 # "Bathtub (GC-fes_min, diy)",
                 # "Bathtub (GC-fes_mean, diy)",
                 # "Bathtub (GC-fes_max, diy)",
                 "Bathtub (CoDEC, aq)",
                 "Bathtub (GC-fes_no, aq)",
                 "Bathtub (GC-fes_min, aq)",
                 "Bathtub (GC-fes_mean, aq)",
                 "Bathtub (GC-fes_max, aq)",
             ]),
             "coastal": widgets.ToggleButton(description="Coastal mask"),
             "compare": widgets.Dropdown(options=[
                 ("No compare", "none"),
                 ("Compare w/o pluvial", "without"),
                 ("Compare w. pluvial from DEM", "w_dempixels"),
                 ("Compare o. pluvial from DEM", "o_dempixels"),
                 ("Compare w. pluvial from catchments", "w_catchments"),
                 ("Compare o. pluvial from catchments", "o_catchments"),
                 # ("Compare w. CaMa (ISIMIP2a)", "w_isimip2a"),
                 # ("Compare o. CaMa (ISIMIP2a)", "o_isimip2a"),
                 ("Compare w. CaMa (ISIMIP3a, no)", "w_isimip3anoprot"),
                 ("Compare w. CaMa (ISIMIP3a, 2y)", "w_isimip3a2yprot"),
                 ("Compare w. CaMa (ISIMIP3a, flopros)", "w_isimip3aflopros"),
                 ("Compare o. CaMa (ISIMIP3a, no)", "o_isimip3anoprot"),
                 ("Compare o. CaMa (ISIMIP3a, 2y)", "o_isimip3a2yprot"),
                 ("Compare o. CaMa (ISIMIP3a, flopros)", "o_isimip3aflopros"),
                 ("Compare bathtub (CLIMADA)", "bt_climada"),
                 # ("Compare bathtub (CoDEC, diy)", "bt_codec"),
                 # ("Compare bathtub (GC-fes_no, diy)", "bt_geoclaw-fes_no"),
                 # ("Compare bathtub (GC-fes_min, diy)", "bt_geoclaw-fes_min"),
                 # ("Compare bathtub (GC-fes_mean, diy)", "bt_geoclaw-fes_mean"),
                 # ("Compare bathtub (GC-fes_max, diy)", "bt_geoclaw-fes_max"),
                 ("Compare bathtub (CoDEC, aq)", "bt_aq_codec"),
                 ("Compare bathtub (GC-fes_no, aq)", "bt_aq_geoclaw-fes_no"),
                 ("Compare bathtub (GC-fes_min, aq)", "bt_aq_geoclaw-fes_min"),
                 ("Compare bathtub (GC-fes_mean, aq)", "bt_aq_geoclaw-fes_mean"),
                 ("Compare bathtub (GC-fes_max, aq)", "bt_aq_geoclaw-fes_max"),
             ]),
             "rainfll": widgets.Dropdown(options=[
                 "No rainfall",
                 "ERA5 rainfall",
                 "ERA5 surface runoff",
                 "Pluv. flood by DEM",
                 "Pluv. flood by catchments",
                 # "CaMa (ISIMIP2a)",
                 "CaMa (ISIMIP3a, no)",
                 "CaMa (ISIMIP3a, 2y)",
                 "CaMa (ISIMIP3a, flopros)",
                 "Water occurrence",
             ]),
             "terrain": widgets.Dropdown(options=[
                 "No terrain data",
                 "Elevation model (DEM)",
                 "Catchments",
             ]),
             "tides": widgets.Dropdown(options=[
                 ("No tides", "none"),
                 ("Tide gauges", "gauges"),
                 ("USGS high water marks", "hwm"),
             ], value="none"),
             "gc_regions": widgets.Dropdown(options=[("No GC regions", "none")], value="none"),
        }
        for w in self.w_select_map_type.values():
            w.observe(lambda change: self.update_map(self.w_select_map.value), names='value')
        for key, w in self.w_select_map_type.items():
            w.layout.width = "18ex" if isinstance(w, widgets.Dropdown) else "16ex"

        self.w_toggle_gauge_tides = widgets.ToggleButton(description="Display tides", value=False)
        self.w_toggle_gauge_tides.observe(lambda change: self.select_gauge(), names="value")

        self.w_select_gauge = widgets.Select(options=[], rows=30)
        self.w_select_gauge.layout.width = "45ex"
        self.w_select_gauge.observe(lambda change: self.select_gauge(), names='value')

        self.w_select_hwm = widgets.Select(options=[], rows=18)
        self.w_select_hwm.layout.width = "30ex"
        self.w_select_hwm.observe(lambda change: self.select_hwm(), names='value')

        self.w_output_worldmap = widgets.Output()
        self.w_output_worldmap.layout.width = "10.5in"
        self.w_output_trackmap = widgets.Output()
        self.w_output_trackmap.layout.width = "4.5in"
        self.w_output_map = widgets.Output()
        self.w_output_map.layout.width = "12in"
        self.w_output_tides = widgets.Output()
        self.w_output_tides.layout.width = "13in"
        self.w_output_hwm = widgets.Output()
        self.w_output2 = widgets.Output()

        with self.w_output_worldmap, plt.ioff():
            fig = plt.figure(figsize=(10, 4), constrained_layout=True)
            self.plot_overview = u_plot.PlotEventsOverview(fig, [], proj_data=self.proj_data)
            display(fig.canvas)

        with self.w_output_trackmap, plt.ioff():
            fig = plt.figure(figsize=(4, 4), constrained_layout=True)
            self.plot_event = u_plot.PlotSingleEvent(fig)
            display(fig.canvas)

        with self.w_output_map, plt.ioff():
            fig = plt.figure(figsize=(10, 4.5), constrained_layout=True)
            self.plot_map = u_plot.PlotMap(fig)
            display(fig.canvas)

        with self.w_output_tides, plt.ioff():
            fig = plt.figure(figsize=(11, 7), constrained_layout=True)
            self.plot_tide = u_plot.PlotTide(fig)
            display(fig.canvas)

        self.plot_overview.figure.canvas.header_visible = False
        self.plot_event.figure.canvas.header_visible = False
        self.plot_map.figure.canvas.header_visible = False
        self.plot_tide.figure.canvas.header_visible = False

        def pick_track_cb(i_tr):
            self.w_select_event.value = i_tr
        self.plot_overview.pick_callbacks.append(pick_track_cb)

        def pick_map_cb(i_map):
            self.w_select_map.value = i_map
        self.plot_event.pick_callbacks.append(pick_map_cb)

        def pick_gauge_cb(i_gauge):
            if self.w_select_map_type['tides'].value == "gauges":
                self.w_select_gauge.value = i_gauge
            else:
                self.w_select_hwm.value = self.w_select_hwm.options[i_gauge[1]][1]
        self.plot_map.pick_callbacks.append(pick_gauge_cb)

        self.w_hbox_gauges = widgets.HBox([
            widgets.VBox([self.w_toggle_gauge_tides, self.w_select_gauge]),
            self.w_output_tides
        ])
        self.w_hbox_hwm = widgets.HBox([self.w_select_hwm, self.w_output_hwm])

        self.update_datareader()

        self.widget = widgets.VBox([
            widgets.HBox([self.w_fm_source, self.w_zos, self.w_toggle_compare_areas]),
            widgets.HBox([
                widgets.VBox([self.w_year_range, self.w_output_worldmap]),
                widgets.VBox([self.w_select_event, self.w_output_trackmap]),
            ]),
            widgets.HBox([
                self.w_select_map,
                widgets.VBox([
                    widgets.HBox(list(self.w_select_map_type.values())),
                    self.w_output_map
                ]),
            ]),
            self.w_hbox_gauges,
            self.w_hbox_hwm,
            self.w_output2,
        ])


    def update_datareader(self):
        self.reader = u_io.DataReader(
            source=self.w_fm_source.value,
            zos=self.w_zos.value,
            compare_areas=self.w_toggle_compare_areas.value,
        )
        self.fm_meta = self.reader.fm_meta

        tracks = [
            np.stack([tr.lon.values, tr.lat.values], axis=1)
            for tr in self.reader.tracks]

        rectangles = []
        for tr in self.reader.tracks:
            tr_rects = []
            for idx, row in self.fm_meta[self.fm_meta['ibtracs_id'] == tr.sid].iterrows():
                tr_rects.append([
                    (row['xmin'], row['ymin']),
                    row['xmax'] - row['xmin'],
                    row['ymax'] - row['ymin'],
                ])
            rectangles.append(tr_rects)

        self.plot_overview.replace_trackset(list(zip(tracks, rectangles)))

        self.filter_tracks()
        self.w_select_map.value = 0


    def update_map(self, i_map):
        row = self.fm_meta.iloc[i_map]
        map_id = row['map_id']
        ibtracs_id = row['ibtracs_id']

        bounds = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        shape = (abs(row['height']), abs(row['width']))
        extent = (bounds[0], bounds[2], bounds[1], bounds[3])

        images = []
        if self.w_select_map_type['compare'].value != "none":
            images.append(self.reader.compare(
                map_id, bounds=bounds, pluvial=self.w_select_map_type['compare'].value))
        else:
            if "DEM" in self.w_select_map_type['terrain'].value:
                images.append(self.reader.elevation(map_id, bounds=bounds, shape=shape))
            elif self.w_select_map_type['terrain'].value == "Catchments":
                images.append(self.reader.catchments(map_id, bounds=bounds, shape=shape))

            source = self.w_select_map_type['reference'].value
            if source != "none":
                images.append(self.reader.floodmap(
                    map_id, source=source, bounds=bounds, shape=shape))

            model_inun = self.w_select_map_type['model_inun'].value
            if model_inun != "No inundation model":
                if model_inun == "GC fgmax":
                    images.append(self.reader.geoclaw(ibtracs_id, bounds, shape))
                elif "bathtub" in model_inun.lower():
                    # extract value in parenthesis
                    mode = model_inun[9:-1]
                    images.append(self.reader.bathtub(
                        map_id, bounds=bounds, shape=shape, mode=mode))

            mode = self.w_select_map_type['rainfll'].value
            if mode != "No rainfall":
                images.append(self.reader.rainfall(map_id, bounds, shape, mode=mode))

            if self.w_select_map_type['coastal'].value:
                images.append(self.reader.coastal_mask(bounds, shape))
        self.plot_map.plot(extent, images, proj_data=self.proj_data)

        if self.w_select_map_type['tides'].value != "none":
            if self.w_select_map_type['tides'].value == "gauges":
                gdata = self.reader.gaugedata(map_id)
                for gsrc in ALL_GAUGE_SOURCES:
                    gcolor = GAUGE_SOURCE_COLORS[gsrc]
                    self.plot_map.add_gauges(
                        np.array([g['location'][::-1] for g in gdata[gsrc]]).reshape(-1, 2),
                        dict(color=GAUGE_SOURCE_COLORS[gsrc], marker=GAUGE_SOURCE_MARKERS[gsrc]),
                        [False if g['geoclaw'] is None
                         else all(h > 0.1 for h in g['geoclaw']['topo_height'])
                         for g in gdata[gsrc]])
            else:
                marks = self.reader.hwm_for_map(map_id)
                self.plot_map.add_gauges(
                    np.array([(m['longitude'], m['latitude']) for m in marks]).reshape(-1, 2),
                    dict(color="orange", edgecolors="black", linewidths=1, marker="D", s=10),
                    [False for m in marks])

        gc_region_sel = self.w_select_map_type['gc_regions'].value
        if gc_region_sel != "none":
            gc_regions = []
            for gc_run in self.reader.gc_regions(ibtracs_id):
                date = gc_region_sel["date"]
                if date is not None and gc_run["date"] != date:
                    continue
                res = gc_region_sel["resolution"]
                for reg in gc_run["regions"]:
                    if res is not None and reg["resolution"] != res:
                        continue
                    gc_regions.append((
                        reg["bounds"][:2],
                        reg["bounds"][2] - reg["bounds"][0],
                        reg["bounds"][3] - reg["bounds"][1],
                    ))
            self.plot_map.add_areas(gc_regions)


    def update_single_tc(self):
        i_tr = self.w_select_event.value
        ibtracs_id = self.reader.tracks[i_tr].attrs['sid']
        tr_maps = self.fm_meta[self.fm_meta['ibtracs_id'] == ibtracs_id]

        # extent of flood maps:
        rectangles = [
            (row['i_map'], [(row['xmin'], row['ymin']), row['xsize'], row['ysize']])
            for idx, row in tr_maps.iterrows()]

        self.plot_event.plot(
            np.stack([self.reader.tracks[i_tr].lon.values,
                      self.reader.tracks[i_tr].lat.values], axis=1),
            rectangles,
            proj_data=self.proj_data)
        self.plot_event.highlight(tr_maps['i_map'].values == self.w_select_map.value)

        gc_regions = self.reader.gc_regions(ibtracs_id)
        self.plot_event.add_areas([
            (
                r["bounds"][:2],
                r["bounds"][2] - r["bounds"][0],
                r["bounds"][3] - r["bounds"][1],
            )
            for gc_runs in gc_regions
            for r in gc_runs["regions"]
            if r["resolution"] == "low"
        ])
        options = [
            ("No GC regions", "none"),
            ("All dates and levels", {"resolution": None, "date": None}),
        ] + [
            (f"All dates: {res}-res", {"resolution": res, "date": None})
            for res in ["low", "med", "hi"]
        ]
        for gc_run in gc_regions:
            options.append(
                (f"{gc_run['date']}: All levels", {"resolution": None, "date": gc_run["date"]})
            )
            for res in ["low", "med", "hi"]:
                options.append(
                    (f"{gc_run['date']}: {res}-res", {"resolution": res, "date": gc_run["date"]})
                )
        self.w_select_map_type['gc_regions'].options = options
        self.w_select_map_type['gc_regions'].value = "none"


    def select_map(self):
        i_map = self.w_select_map.value
        i_tr = self.fm_meta['i_track'].iloc[i_map]
        map_id = self.fm_meta['map_id'].iloc[i_map]

        if self.w_select_event.value == i_tr:
            ibtracs_id = self.reader.tracks[i_tr].attrs['sid']
            tr_maps = self.fm_meta[self.fm_meta['ibtracs_id'] == ibtracs_id]
            self.plot_event.highlight(tr_maps['i_map'].values == self.w_select_map.value)
        else:
            self.w_select_event.value = i_tr
        self.update_map(i_map)

        gdata = self.reader.gaugedata(map_id, geoclaw=False)
        if all(len(gdata[gsrc]) == 0 for gsrc in ALL_GAUGE_SOURCES):
            self.w_hbox_gauges.layout.display = "none"
        else:
            self.w_hbox_gauges.layout.display = None

        options = []
        for i_gsrc, gsrc in enumerate(ALL_GAUGE_SOURCES):
            labels = [
                f"{gsrc} ({g['location'][0]:.1f}/{g['location'][1]:.1f}): {g['filename']}"
                for g in gdata[gsrc]]
            options.extend(
                (g, (i_gsrc, i_gauge)) for i_gauge, g in enumerate(labels)
            )
        self.w_select_gauge.options = options

        hwm_data = self.reader.hwm_for_map(map_id)
        if len(hwm_data) == 0:
            self.w_hbox_hwm.layout.display = "none"
        else:
            self.w_hbox_hwm.layout.display = None
        self.w_select_hwm.options = [
            (f"HWM #{m['hwm_id']} ({m['longitude']:.3f}/{m['latitude']:.3f})", m['hwm_iter'])
            for m in hwm_data
        ]


    def select_gauge(self):
        i_gauge = self.w_select_gauge.value
        if i_gauge is None:
            self.plot_tide.figure.clf()
            return
        i_map = self.w_select_map.value
        map_id = self.fm_meta['map_id'].iloc[i_map]
        gdata = self.reader.gaugedata(map_id)

        gsrc = ALL_GAUGE_SOURCES[i_gauge[0]]
        gdata = gdata[gsrc][i_gauge[1]]
        title = f"{gdata['filename']} ({gdata['location'][0]:.3f}/{gdata['location'][1]:.3f})"
        ref_label = f"{gsrc} surge"

        ref_plots = [
            (gdata['referenced'], dict(color=GAUGE_SOURCE_COLORS[gsrc], label=ref_label)),
        ]

        if self.w_toggle_gauge_tides.value:
            st_msl = gdata['annual_msl']
            fes_series = gdata['combined'].copy()
            fes_series.values[:] = u_gauge.compute_fes_tides(
                *gdata['gc_location'][::-1],
                fes_series.index.values.astype('datetime64[us]'),
                ref_annual_msl=False,
            )
            ref_plots.extend([
                (gdata['combined'] - st_msl, dict(color="tab:red", label="combined")),
                (gdata['tide_levels'] - st_msl, dict(color="tab:purple", label="tides")),
                (gdata['tide_levels_full'] - st_msl, dict(color="skyblue", label="tides full")),
                (fes_series, dict(color="tab:orange", label="FES2014")),
            ])

        reference_sl = 0
        if gdata['geoclaw'] is not None and np.isfinite(gdata['geoclaw']['annual_msl']):
            reference_sl = gdata['geoclaw']['annual_msl']

        t_pad = np.timedelta64(12, 'h')
        t_start = t_end = None
        all_plots = []
        if gdata['geoclaw'] is not None:
            if len(gdata['geoclaw']['time']) > 0:
                all_plots.extend([
                    (sl, dict(color="blue", label="geoclaw"))
                    for sl in gdata['geoclaw']['referenced']
                ])
                t_start = np.amin([t[0] for t in gdata['geoclaw']['time']]) - t_pad
                t_end = np.amax([t[-1] for t in gdata['geoclaw']['time']]) + t_pad

        if t_start is None and len(ref_plots) > 0:
            t_start = np.amin([gseries.index[0] for gseries, _ in ref_plots])
            t_end = np.amax([gseries.index[-1] for gseries, _ in ref_plots])

        for ref_plot in ref_plots:
            gseries = ref_plot[0]
            gseries = gseries[(gseries.index >= t_start) & (gseries.index <= t_end)]
            if np.isfinite(gseries).sum() > 2:
                all_plots.append((gseries, ref_plot[1]))

        # do the plotting with annotations
        self.plot_tide.plot(all_plots, title=title)
        if gdata['geoclaw'] is not None:
            all_base_sl = gdata['geoclaw']['base_sea_level']
            all_topo_heights = gdata['geoclaw']['topo_height']
            times = gdata['geoclaw']['time']
            for base, h, idx in zip(all_base_sl, all_topo_heights, times):
                self.plot_tide.annotate(
                    "base sea level", (idx[0], (base - reference_sl) * 1000), yoffset=0,
                )
                self.plot_tide.annotate(
                    "topo height", (idx[0], (h - reference_sl) * 1000), yoffset=10,
                )

        wind_kwargs = dict(color='tab:green', linestyle="--", label="wind speed")
        if gdata['wind'] is not None:
            wseries = gdata['wind'].intensity
            if t_start is not None:
                wseries = wseries[(wseries.index >= t_start) & (wseries.index <= t_end)]
            wseries = wseries[wseries > 0]
            self.plot_tide.plot_twin(
                [(wseries, wind_kwargs)],
                ylabel="wind speed (m/s)",
                ylim=(0, 100),
                yscale=u_const.SAFFIR_SIMPSON_YSCALE,
            )


    def select_hwm(self):
        hwm_iter = self.w_select_hwm.value
        if hwm_iter is None:
            return
        hwm = self.reader.high_water_marks[hwm_iter]

        i_map = self.w_select_map.value
        map_id = self.fm_meta['map_id'].iloc[i_map]
        fes = self.reader.zos.split("-fes_")[1]

        hwm_keys = [
            "latitude", "longitude", "height_above_gnd_m", "elev_m",
            "dem", f'geoclaw-fes_{fes}', f'bt_aq_geoclaw-fes_{fes}', f'bt_aq_codec',
            "hwmQualityName", "hwm_uncertainty", "uncertainty", "hwmTypeName", "markerName",
            "hwm_environment", "hwm_locationdescription", "hwm_notes", "siteDescription",
            "survey_date", "flag_date", "eventName", "verticalDatumName", "verticalMethodName",
            "file_urls"
        ]

        locations = [(hwm['longitude'], hwm['latitude'])]
        for bt_mode in [f"geoclaw-fes_{fes}", "codec"]:
            hwm[f'bt_aq_{bt_mode}'] = self.reader.bathtub_sample(
                map_id, locations, mode=f"{bt_mode},aq",
            )[0]
        hwm[f'geoclaw-fes_{fes}'] = self.reader.geoclaw_sample(hwm['ibtracs_id'], locations)[0]
        hwm['dem'] = self.reader.dem_sample(locations)[0]

        hwm_keys = [k for k in hwm_keys if k in hwm.keys()]
        self.w_output_hwm.clear_output(wait=True)
        with self.w_output_hwm:
            pd.set_option('display.max_colwidth', None)
            df = pd.DataFrame(
                [hwm[k] for k in hwm_keys],
                index=hwm_keys,
                columns=[f"HWM #{hwm['hwm_id']} ({hwm['longitude']:.3f}/{hwm['latitude']:.3f})"])
            df.loc['file_urls',:] = df.loc['file_urls',:].apply(
                lambda pairs: ', '.join(f'<a target="_blank" href="{url}">{name}</a>' for name, url in pairs))
            display(
                df.style
                .set_table_styles([
                    {'selector': 'td', 'props': 'width:1000px;'},
                ])
                .format(lambda v: v if isinstance(v, str) else f'{float(v):.2f}')
           )


    def filter_tracks(self):
        yr_start, yr_end = self.w_year_range.value
        track_mask = [yr_start <= int(tr.attrs['sid'][:4]) <= yr_end
                      for tr in self.reader.tracks]
        self.plot_overview.filter(track_mask)

        sel_tracks = np.asarray(track_mask).nonzero()[0]
        i_tr_sel = 0
        if self.w_select_event.value is not None:
            i_tr_sel = int(np.clip(self.w_select_event.value, sel_tracks[0], sel_tracks[-1]))
        self.w_select_event.options = [
            (f"{self.reader.tracks[i_tr].attrs['sid']} "
             f"({self.reader.tracks[i_tr].attrs['ibtracs_category']} "
             f"{self.reader.tracks[i_tr].attrs['ibtracs_name']})", i_tr)
            for i_tr in sel_tracks
        ]
        self.w_select_event.value = i_tr_sel

        sel_maps = self.fm_meta[np.isin(self.fm_meta['i_track'], sel_tracks)]
        i_map_sel = sel_maps[sel_maps['i_track'] == i_tr_sel]['i_map'].values[0]
        self.w_select_map.options = list(zip(
            sel_maps.apply(lambda row: (
                f"{row['map_id']} "
                f"{'📈' if row['has_gauges'] else '⬜'}"
                f"{'📏' if row['has_hwm'] else '⬜'}"
                ), axis='columns'),
            sel_maps['i_map']))
        self.w_select_map.value = i_map_sel


