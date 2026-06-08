
import json
import re
import warnings

from climada.hazard import TCTracks
from climada.util import log_level
import dask
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import xarray as xr

import gcvalid.plot.common as u_plot
from gcvalid.util import dt64_to_dmy
import gcvalid.util.constants as u_const
import gcvalid.util.gauge as u_gauge


def read_ds_with_date_and_bounds(ds, var, date, bounds):
    if "longitude" in ds.variables:
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})

    # pad to make sure we don't lose values close to the boundary
    pad_lon = 0.5 * abs(ds["lon"].values[1] - ds["lon"].values[0])
    pad_lat = 0.5 * abs(ds["lat"].values[1] - ds["lat"].values[0])
    bounds = (
        bounds[0] - pad_lon,
        bounds[1] - pad_lat,
        bounds[2] + pad_lon,
        bounds[3] + pad_lat,
    )

    if bounds[0] < 0 and ds["lon"].values.min() >= 0:
        bounds = (bounds[0] + 360, bounds[1], bounds[2] + 360, bounds[3])

    for i, dim in enumerate(["lon", "lat"]):
        [idx] = ((bounds[i] <= ds[dim]) & (ds[dim] <= bounds[i + 2])).values.nonzero()
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            ds = ds.isel(indexers={dim: slice(idx[0], idx[-1] + 1)})

    lat, lon = ds["lat"].values, ds["lon"].values
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]
    if lon[0] > 180:
        lon -= 360
    transform = rasterio.Affine(
        dlon, 0, lon[0] - 0.5 * dlon,
        0, dlat, lat[0] - 0.5 * dlat,
    )
    shape = (ds.dims['lat'], ds.dims['lon'])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Converting non-nanosecond precision datetime values to nanosecond precision",
            category=UserWarning,
        )
        ds = ds.sel(time=(ds["time"].astype(date.dtype) == date))

    if ds.dims["time"] == 0:
        data = np.full(shape, np.nan, dtype=dtype)
    else:
        data = ds[var].compute().values[0, :, :]

    return data, transform


def read_nc_with_date_and_bounds(path, var, date, bounds):
    with xr.open_dataset(path) as ds:
        data, transform = read_ds_with_date_and_bounds(ds, var, date, bounds)
    return data, transform


def read_daily_rainfall(source, date, bounds, spatial_max=True, sro=False):
    if source == "gpcc":
        day, month, year = dt64_to_dmy(date)
        path = u_const.GPCC_PATH / u_const.GPCC_FNAME.format(year=year)
        if sro:
            raise NotImplementedError
    elif source == "wfde5":
        day, month, year = dt64_to_dmy(date)
        path = u_const.WFDE5_PATH / u_const.WFDE5_FNAME.format(year=year, month=month)
        if sro:
            raise NotImplementedError

        # WFDE5 does not cover the whole required time range, return NaN outside
        if not path.exists():
            return np.nan if spatial_max else (None, None)
    elif source == "era5l":
        # in ERA5-Land, rainfall at 00:00:00 is accumulated value from previous day
        date += np.timedelta64(1, "D")
        day, month, year = dt64_to_dmy(date)
        fname = u_const.ERA5L_SRO_FNAME if sro else u_const.ERA5L_PR_FNAME
        path = u_const.ERA5L_PATH / fname.format(year=year, month=month)
    elif source == "era5":
        fname = u_const.ERA5_SRO_FNAME if sro else u_const.ERA5_PR_FNAME
        path = u_const.ERA5_PATH / fname
    elif source == "era5_combined":
        return read_daily_rainfall_era5_combined(date, bounds, spatial_max=spatial_max, sro=sro)
    else:
        raise NotImplementedError

    # make sure the date is in daily resolution
    date = date.astype("datetime64[D]")

    var = {
        "gpcc": "precip",
        "wfde5": "Rainf",
        "era5l": "sro" if sro else "tp",
        "era5": "sro" if sro else "pr",
    }[source]

    rainf, transform = read_nc_with_date_and_bounds(path, var, date, bounds)

    if source in ["era5", "era5l"]:
        # convert from m to mm
        rainf *= 1000

    if not spatial_max:
        return rainf, transform
    return np.nanmax(rainf)


def read_daily_rainfall_era5_combined(date, bounds, spatial_max=True, sro=False):
    rainf, transform = read_daily_rainfall("era5", date, bounds, spatial_max=False, sro=sro)
    rainf_l, transform_l = read_daily_rainfall("era5l", date, bounds, spatial_max=False, sro=sro)

    rainf_fine = np.zeros_like(rainf_l)
    rasterio.warp.reproject(
        source=rainf,
        destination=rainf_fine,
        src_transform=transform,
        src_crs=u_const.DEFAULT_CRS,
        dst_transform=transform_l,
        dst_crs=u_const.DEFAULT_CRS,
        resampling=rasterio.warp.Resampling.nearest)

    nan_mask = np.isnan(rainf_l)
    rainf_l[nan_mask] = rainf_fine[nan_mask]
    if not spatial_max:
        return rainf_l, transform_l
    return np.nanmax(rainf_l)


def read_raster_reproject(path, transform=None, path_transform=None, shape=None, dtype=None,
                          resampling="nearest"):
    if isinstance(resampling, str):
        resampling = getattr(rasterio.warp.Resampling, resampling)

    crs = None
    if path_transform is not None:
        with rasterio.open(path_transform, "r") as dst:
            shape = (dst.height, dst.width)
            transform = dst.transform
            crs = dst.crs

    with rasterio.open(path, "r") as src:
        if dtype is None:
            dtype = src.dtypes[0]

        if crs is None:
            crs = src.crs

        data_dst = np.zeros(shape, dtype=dtype)

        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=data_dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=resampling)
    return data_dst


def write_raster(path_out, data, transform=None, path_transform=None,  shape=None, nodata=None):
    if shape is None:
        shape = data.shape

    crs = u_const.DEFAULT_CRS
    if path_transform is not None:
        with rasterio.open(path_transform, "r") as src:
            crs = src.crs
            transform = src.transform

    dst_kwargs = {
        "driver": "GTiff",
        "compress": "deflate",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        dst_kwargs["nodata"] = nodata
    print(f"Writing to {path_out}...")
    with rasterio.open(path_out, "w", **dst_kwargs) as dst:
        dst.write(data.reshape(shape), 1)


def read_compare_df(
    source, pluvial, zos, model_thresh, apply_filters=False, verbose=False, filter_by_area=None,
):
    compare_dir = u_const.COMPARE_DIR / source / pluvial / zos

    zos_names = [zos]
    if zos == "bestcsi":
        zos_names = [f"{zos}-fes_{fes}"
                     for fes in ["min", "mean", "max", "no"]
                     for zos in ["0", "aviso", "mercator"]]
    elif zos.endswith("-fes_bestcsi"):
        zos_names = [f"{zos.split('-fes_')[0]}-fes_{fes}"
                     for fes in ["min", "mean", "max"]]

    compare_dfs = []
    for zos in zos_names:
        df = pd.read_csv(compare_dir / f"stats-thresh_{model_thresh:.1f}.csv")
        df['zos'] = zos
        df["pluvial"] = pluvial
        df["source"] = source
        df["coastal_fm+_area"] = df["coastal_fm_area"] + df["coastal_both_area"]
        compare_dfs.append(df.sort_values(by="map_id").reset_index(drop=True))
    compare_ds = pd.concat(compare_dfs).set_index(["zos", "map_id"]).to_xarray()
    msk_valid = (compare_ds['coastal_fm+_area'] > 0).any(dim="zos")
    n_noarea = (~msk_valid).values.sum()
    if verbose:
        print(f"{n_noarea} events discarded due to no flooded area.")
    compare_ds = compare_ds.sel(map_id=msk_valid)
    compare_df = (
        compare_ds
        .isel(compare_ds['coastal_flooded_both_p'].argmax(dim=["zos"], skipna=True))
        .to_dataframe()
        .dropna(subset=["lon_min"])
        .reset_index()
    )

    compare_df['dem_comment'] = df['dem_comment'].fillna("")
    compare_df['lon_mean'] = 0.5 * (compare_df["lon_min"] + compare_df["lon_max"])
    compare_df['lat_mean'] = 0.5 * (compare_df["lat_min"] + compare_df["lat_max"])
    compare_df['width'] = compare_df['lon_max'] - compare_df['lon_min']
    compare_df['height'] = compare_df['lat_max'] - compare_df['lat_min']
    compare_df['ibtracs_id'] = compare_df.map_id.str.slice(0, 13)

    wind_mask = (compare_df['maxwind'] > 0)
    if not wind_mask.all():
        print(f"{(~wind_mask).sum()} events have maxwind==0!")

    if not apply_filters:
        return compare_df

    if filter_by_area is not None:
        area_mask = (
            (compare_df['coastal_fm_area'] >= filter_by_area[0])
            & (compare_df['coastal_fm_area'] <= filter_by_area[1])
        )
        compare_df = compare_df[area_mask]

    if pluvial.startswith("bt_aq_"):
        # remove events in 2019 because there is no CoDEC data for that year
        yr_mask = ~compare_df['ibtracs_id'].str.startswith("2019")
        compare_df = compare_df[yr_mask]

    if verbose:
        if filter_by_area is not None:
            print(f"{(~area_mask).values.sum()} events discarded due to area filter.")
        if "codec" in pluvial and not yr_mask.all():
            print(f"{(~yr_mask).values.sum()} events discarded due to year 2019.")

    return compare_df


def _set_compare_areas(source, zos, fm_meta):
    fm_meta = fm_meta.reset_index(drop=True)
    compare_dfs = [
        read_compare_df(
            source, "without", zos, 0.0, apply_filters=False, verbose=False,
        ).sort_values(by=["map_id"]).reset_index(drop=True)
    ]
    compare_df = compare_dfs[0].copy()
    for coord in ["lon", "lat"]:
        for op in ["min", "max"]:
            col = f"{coord}_{op}"
            newcol = f"{'x' if coord == 'lon' else 'y'}{op}"
            compare_df[newcol] = getattr(np, f"a{op}")(
                [df[col].values for df in compare_dfs], axis=0)
    cols = ["xmin", "ymin", "xmax", "ymax"]
    compare_df = (
        fm_meta[["map_id", "xres", "yres"]]
        .merge(compare_df[["map_id"] + cols], on="map_id", how="left")
        .reset_index(drop=True)
    )
    for coord, col in {"x": "width", "y": "height"}.items():
        compare_df[col] = np.abs(np.round(
            (compare_df[f"{coord}max"] - compare_df[f"{coord}min"])
            / compare_df[f"{coord}res"]
        )).fillna(0).astype(int)

    zero_mask = (compare_df["width"] * compare_df["height"]) == 0
    compare_df.loc[zero_mask, cols + ["width", "height"]] = np.nan

    finite_mask = np.isfinite(compare_df[cols]).all(axis=1)
    fm_meta.loc[finite_mask, cols + ["width", "height"]] = (
        compare_df.loc[finite_mask, cols + ["width", "height"]]
    )

    return fm_meta


def read_raster_with_bounds(path, bounds=None, shape=None, resampling=None):
    if resampling is None:
        resampling = rasterio.warp.Resampling.bilinear

    with rasterio.open(path, "r") as src:
        if bounds is None:
            transform, width, height = src.transform, src.width, src.height
            if src.crs != u_const.DEFAULT_CRS:
                transform, width, height = rasterio.warp.calculate_default_transform(
                    src.crs, u_const.DEFAULT_CRS, src.width, src.height, *src.bounds)
            shape = (height, width)
        else:
            if shape is None:
                if src.crs != u_const.DEFAULT_CRS:
                    raise NotImplementedError(
                        "Cannot guess destination's shape when CRS changes."
                    )
                res = (np.abs(src.transform[0]), np.abs(src.transform[4]))
                width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
                shape = (int(round(height / res[1])), int(round(width / res[0])))
            transform = rasterio.transform.from_bounds(*bounds, shape[1], shape[0])

        data = np.zeros(shape + (src.count,), dtype=np.float64)
        for iband in range(src.count):
            band_data = np.zeros(data.shape[:-1], dtype=data.dtype)
            rasterio.warp.reproject(
                source=rasterio.band(src, iband + 1),
                destination=band_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=u_const.DEFAULT_CRS,
                dst_nodata=np.nan,
                resampling=resampling,
            )
            data[:, :, iband] = band_data
        if src.count == 1:
            data = data[:, :, 0]
        if src.bounds[1] > src.bounds[3]:
            data = np.flip(data, axis=0)
    return data


class DataReader():
    def __init__(self, source="gfd", zos="aviso-fes_max", compare_areas=True):
        self.source = source
        self.zos = zos
        with log_level(level='ERROR', name_prefix='climada'):
            self.tracks = TCTracks.from_netcdf(u_const.TRACKS_DIR / source).data
        self.tracks = sorted(self.tracks, key=lambda tr: tr.attrs["sid"])

        self.floodmaps_dir = u_const.FLOODMAPS_DIR / source
        self.geoclaw_dir = u_const.GEOCLAW_DIR / source
        self.geoclaw_data_dir = self.geoclaw_dir / "results"
        self.geoclaw_meta_dir = self.geoclaw_dir / "meta"
        self.catchment_dir = u_const.CATCHMENTS_DIR / "windowed" / self.source

        with u_const.HWMS_FILE.open("r") as fp:
            self.high_water_marks = json.load(fp)
        self.fev_events = {d['ibtracs_id']: d['eventName'] for d in self.high_water_marks}

        fm_meta = pd.read_hdf(self.floodmaps_dir / "meta.hdf5").sort_values(by="map_id")
        if compare_areas:
            fm_meta = _set_compare_areas(source, zos, fm_meta)
        fm_meta['xsize'] = fm_meta['xmax'] - fm_meta['xmin']
        fm_meta['ysize'] = fm_meta['ymax'] - fm_meta['ymin']
        fm_meta['i_track'] = fm_meta['ibtracs_id'].apply(
            lambda sid: [i_tr for i_tr, t in enumerate(self.tracks) if t.attrs['sid'] == sid][0])
        fm_meta['i_map'] = np.arange(fm_meta.shape[0])
        fm_meta['has_hwm'] = np.isin(fm_meta.ibtracs_id, list(self.fev_events.keys()))
        self.fm_meta = fm_meta


    def gaugedata(self, map_id, geoclaw=True):
        gdata_by_gsrc = u_gauge.load_gaugedata(
            self.source, map_id, by_gsrc=True, referenced=True,
            geoclaw_zos=self.zos if geoclaw else None)

        if geoclaw:
            for gsrc, stations in gdata_by_gsrc.items():
                for stdata in stations:
                    if stdata['geoclaw'] is not None:
                        [stdata['geoclaw']] = stdata['geoclaw']

        return gdata_by_gsrc


    def floodmap(self, map_id, source=None, bounds=None, shape=None):
        source = self.source if source is None else source

        dirpath = u_const.FLOODMAPS_DIR / (
            "dfo/geotiff_by_sid" if source == "dfo_raw"
            else f"{source}/clean_by_sid"
        )

        paths = (
            [dirpath / f"{map_id}.tif"] if (
                source == self.source
                or source == "dfo_raw" and self.source == "dfo"
            ) else list(dirpath.glob(f"{map_id[:-2]}-*.tif"))
        )

        if len(paths) == 0:
            rgba_data = np.zeros((2, 2, 4))
            rgba_data[:] = np.array([0.8, 0.8, 0.8, 1])[None, None]
            return rgba_data

        with rasterio.Env(VRT_SHARED_SOURCE=False):
            datas = [
                read_raster_with_bounds(
                    path, bounds=bounds, shape=shape,
                    resampling=rasterio.warp.Resampling.nearest)
                for path in paths
            ]

        # combine all maps into one
        data = datas[0]
        for d in datas[1:]:
            nan_mask = np.isnan(data)
            if data.ndim > 2:
                nan_mask = nan_mask.any(axis=-1)
            data[nan_mask] = d[nan_mask]

        # map integers to RGBA colors
        rgba_data = np.zeros(data.shape[:2] + (4,))
        nan_mask = np.isnan(data)
        if source == "dfo":
            nan_mask |= (data == 2)
            rgba_data[data == 1] = np.array([1, 0, 0, 1])[None]
        elif source == "dfo_raw":
            nan_mask = nan_mask.any(axis=-1)
            finite_mask = ~nan_mask
            rgba_data[finite_mask, 3] = 1
            rgba_data[finite_mask, :3] = data[finite_mask, :] / 255
        elif source == "rapid":
            nan_mask |= (data == 255)
            rgba_data[data > 0] = np.array([1, 0, 0, 1])[None]
        elif source == "gfd":
            rgba_data[data == 1] = np.array([1, 0, 0, 1])[None]
        rgba_data[nan_mask] = np.array([0.8, 0.8, 0.8, 1])[None]

        return rgba_data


    def compare(self, map_id, bounds=None, pluvial="without"):
        path = u_const.COMPARE_DIR / self.source / pluvial / self.zos / f"{map_id}-thresh_0.1.tif"
        with rasterio.Env(VRT_SHARED_SOURCE=False):
            data = read_raster_with_bounds(
                path, bounds=bounds, resampling=rasterio.warp.Resampling.nearest)
        return u_plot.compare2rgba(data)


    def geoclaw(self, ibtracs_id, bounds, shape):
        path = self.geoclaw_data_dir / f"{ibtracs_id}_{self.source}-zos_{self.zos}.tif"
        with rasterio.Env(VRT_SHARED_SOURCE=False):
            data = read_raster_with_bounds(path, bounds=bounds, shape=shape)
        rgba_data = np.zeros(shape + (4,))
        rgba_data[data > 0] = np.array([0, 0, 1, 1])[None]
        rgba_data[data > 0, 3] = np.clip(0.2 + 0.8 * data[data > 0] / 5, 0, 1)
        return rgba_data


    def geoclaw_sample(self, ibtracs_id, locations):
        path = self.geoclaw_data_dir / f"{ibtracs_id}_{self.source}-zos_{self.zos}.tif"
        with rasterio.Env(VRT_SHARED_SOURCE=False):
            with rasterio.open(path, "r") as src:
                return [v[0] for v in src.sample(locations)]


    def bathtub_sample(self, map_id, locations, mode="CLIMADA"):
        # CLIMADA, CoDEC, GC-fes_(no|min|mean|max)
        mode = mode.lower().replace("gc-", "geoclaw-")
        mode_aq = False
        if "," in mode:
            mode, mode_aq = mode.split(",")
            mode_aq = mode_aq.strip() == "aq"

        path = u_const.BATHTUB_DIR / mode / self.source / (
            f"aqueduct_output/{map_id}/inun.tif"
            if mode_aq
            else f"{map_id}.tif"
        )
        with rasterio.Env(VRT_SHARED_SOURCE=False):
            with rasterio.open(path, "r") as src:
                return [v[0] for v in src.sample(locations)]


    def dem_sample(self, locations):
        with rasterio.Env(VRT_SHARED_SOURCE=False):
            with rasterio.open(u_const.DEM_FILE, "r") as src:
                return [v[0] for v in src.sample(locations)]


    def coastal_mask(self, bounds, shape):
        with rasterio.Env(VRT_SHARED_SOURCE=False):
            data_water = read_raster_with_bounds(
                u_const.WATERBODY_FILE, bounds=bounds, shape=shape)
            data_dem = read_raster_with_bounds(
                u_const.DEM_FILE, bounds=bounds, shape=shape,
                resampling=rasterio.warp.Resampling.average)
        rgba_data = np.zeros(shape + (4,))
        # cutoff: water occurrence >5% or height >10m or height <-10m
        mask = (data_water > 5) | (np.abs(data_dem) > 10)
        rgba_data[mask, 3] = 1
        rgba_data[mask, :3] = 0.8
        return rgba_data


    def bathtub(self, map_id, bounds=None, shape=None, mode="CLIMADA"):
        # CLIMADA, CoDEC, GC-fes_(no|min|mean|max)
        mode = mode.lower().replace("gc-", "geoclaw-")
        mode_aq = False
        if "," in mode:
            mode, mode_aq = mode.split(",")
            mode_aq = mode_aq.strip() == "aq"

        with rasterio.Env(VRT_SHARED_SOURCE=False):
            path = u_const.BATHTUB_DIR / mode / self.source / (
                f"aqueduct_output/{map_id}/inun.tif"
                if mode_aq
                else f"{map_id}.tif"
            )
            data = read_raster_with_bounds(
                path, bounds=bounds, shape=shape,
                resampling=rasterio.warp.Resampling.nearest)

        nnz_mask = (data > 0)
        rgba_data = np.zeros(data.shape + (4,))
        rgba_data[nnz_mask] = np.array([0, 0, 1, 1])[None]
        rgba_data[nnz_mask, 3] = np.clip(0.2 + 0.8 * data[nnz_mask] / 5, 0, 1)
        return rgba_data


    def rainfall(self, map_id, bounds, shape, mode="raw"):
        if "DEM" in mode:
            mode = "dempixels"
        elif "catchment" in mode.lower():
            mode = "catchments"
        elif "ISIMIP2a" in mode:
            mode = "isimip2a"
        elif "ISIMIP3a" in mode:
            prot = (
                "flopros" if "flopros" in mode else
                "2yprot" if "2y" in mode else
                "noprot"
            )
            mode = "isimip3a{prot}"
        elif "occurrence" in mode.lower():
            mode = "occurrence"
        elif "runoff" in mode.lower():
            mode = "runoff"
        else:
            mode = "rainfall"

        data = np.zeros(shape)
        with rasterio.Env(VRT_SHARED_SOURCE=False):
            if mode != "occurrence":
                path_rain = u_const.PLUVIAL_MAPS_DIR / mode / self.source
                path = path_rain / f"{map_id}.tif"
                data[:] = read_raster_with_bounds(
                    path, bounds=bounds, shape=shape,
                    resampling=rasterio.warp.Resampling.nearest)
            data_occ = read_raster_with_bounds(
                u_const.WATERBODY_FILE, bounds=bounds, shape=shape)

        if mode == "isimip2a":
            data *= 5
        elif mode == "rainfall":
            data[np.isnan(data)] = 0
            data = np.clip((data - 50) / 60, 0, 5)
        elif mode == "runoff":
            data /= 60

        data = np.fmax(data_occ / 100, data / 5)

        nnz_mask = (data > 1e-2)
        rgba_data = np.zeros(data.shape + (4,))
        rgba_data[nnz_mask] = np.array([0, 0, 1, 1])[None]
        rgba_data[nnz_mask, 3] = np.clip(0.2 + 0.8 * data[nnz_mask], 0, 1)
        return rgba_data


    def catchments(self, map_id, bounds=None, shape=None):
        path = self.catchment_dir / f"{map_id}.tif"
        with rasterio.Env(VRT_SHARED_SOURCE=False):
            data = read_raster_with_bounds(
                path, bounds=bounds, shape=shape,
                resampling=rasterio.warp.Resampling.nearest)
        nnz_mask = (data < 1)
        rgba_data = np.zeros(data.shape + (4,))
        rgba_data[nnz_mask] = np.array([0, 0.6, 0, 1])[None]
        rgba_data[nnz_mask, 3] = np.clip(1 - data[nnz_mask], 0, 1)
        return rgba_data


    def elevation(self, map_id, bounds=None, shape=None):
        path = u_const.ELEVATION_MAPS_DIR / self.source / f"{map_id}.tif"
        with rasterio.Env(VRT_SHARED_SOURCE=False):
            data = read_raster_with_bounds(
                path, bounds=bounds, shape=shape,
                resampling=rasterio.warp.Resampling.nearest)
        cmap, cnorm = u_plot.colormap_coastal_dem()
        rgba_data = cmap(cnorm(data.ravel())).reshape(data.shape + (4,))
        return rgba_data


    def gc_regions(self, ibtracs_id):
        result = []
        fname = f"{ibtracs_id}_{self.source}-zos_{self.zos}-*-regions.data"
        for path in self.geoclaw_meta_dir.glob(fname):
            regions = [
                re.sub(r" +", " ", l).split(" ")
                for l in path.read_text().split("\n")
                if len(l) > 0 and l[0] != "#"
            ]
            regions = [
                # 8 floats: res_min, res_max, t_min, t_max, lon_min, lon_max, lat_min, lat_max
                {
                    "resolution": (int(r[0]), int(r[1])),
                    "bounds": (float(r[4]), float(r[6]), float(r[5]), float(r[7])),
                }
                for r in regions if len(r) >= 8
            ]

            # translate resolutions into low/med/hi scale
            resolutions = sorted(set(r["resolution"] for r in regions))
            for r in regions:
                r["resolution"] = ["low", "med", "hi"][resolutions.index(r["resolution"])]

            date = np.datetime64(re.sub(
                r".*-([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]{2})-regions.data",
                r"\1T\2",
                path.name,
            ))
            result.append({"date": date, "regions": regions})
        return result


    def hwm_for_map(self, map_id):
        row = self.fm_meta[self.fm_meta['map_id'] == map_id].iloc[0]
        if row['ibtracs_id'] not in self.fev_events:
            return []
        fev_name = self.fev_events[row['ibtracs_id']]
        selection = []
        for hwm_iter, hwm in enumerate(self.high_water_marks):
            hwm['hwm_iter'] = hwm_iter
            within_bounds = (
                row['xmin'] <= hwm['longitude'] <= row['xmax']
                and row['ymin'] <= hwm['latitude'] <= row['ymax'])
            has_height_info = hwm['height_above_gnd_m'] > 0 or hwm['elev_m'] > 0
            if within_bounds and hwm['eventName'] == fev_name and has_height_info:
                hwm['file_urls'] = [
                    (f['name'], f'https://stn.wim.usgs.gov//STNServices/Files/{f["file_id"]}/Item')
                    for f in hwm['files']
                ]
                selection.append(hwm)
        return selection
