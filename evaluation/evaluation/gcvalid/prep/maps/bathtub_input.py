"""
Prepare input data for aqueduct bathtub tool:

https://github.com/Deltares/aqueduct-coastal-flooding/blob/py38/

within flood map area at 30 arc-seconds resolution
"""
import argparse
import os

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from gcvalid.prep.maps.elevation import read_raster_from_fm_meta
import gcvalid.util.constants as u_const
import gcvalid.util.gauge as u_gauge
import gcvalid.util.io as u_io


ALTIMETRY_SRC = "aviso"
"""The altimetry data source to use"""

INI_TEMPLATE = """[maps]
dem_file = input/{source}/{map_id}/dem.tif
egm_file = input/{source}/{map_id}/egm.tif
water_perc_file = input/{source}/{map_id}/occurrence.tif

[metadata]
source=Global Tide and Surge Model
institution=Deltares
title=Aqueduct Coastal hazard layer
references=http://floods.wri.org/
Conventions=CF-1.6
project=Aqueduct Global Flood Analyzer

[flood_routine]
tempdir=tmp/inun
waterp_thresh=100.0
"""
"""A template for the configuration file that is used for each run"""


def write_waterlevel(path, locations, waterlevel_m):
    # create new xarray.Dataset with variables for location, time and waterlevel
    ds = xr.Dataset(
        data_vars={
            "waterlevel": (["time", "station"], waterlevel_m[None]),
        },
        coords={
            "station_x_coordinate": ("station", locations[:, 0]),
            "station_y_coordinate": ("station", locations[:, 1]),
        }
    )

    print(f"Writing to {path} ...")
    encoding = {v: {"zlib": True} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)


def restrict_to_wind_exposure(stdata):
    stdata['combined_exp'] = None
    stdata["geoclaw_exp"] = None
    stdata["wind_period"] = (None, None)

    if stdata["wind"] is None:
        return False

    # extract temporal range (from first till last time of wind exposure ±12 hours)
    idx_exposed = (stdata['wind'].intensity.values >= 17.5).nonzero()[0]
    if idx_exposed.size < 2:
        return False
    t_start, t_end = stdata['wind'].index[idx_exposed[[0, -1]]]
    t_pad = np.timedelta64(12, 'h')
    t_start, t_end = (t_start - t_pad, t_end + t_pad)
    stdata['wind_period'] = (t_start, t_end)

    t_mask = (stdata['combined'].index >= t_start) & (stdata['combined'].index <= t_end)
    stdata['combined_exp'] = stdata['combined'][t_mask].dropna()
    if stdata["combined_exp"].size == 0:
        stdata["combined_exp"] = None

    if "geoclaw" in stdata:
        if stdata["geoclaw"] is None or len(stdata["geoclaw"][0]['referenced']) == 0:
            return True

        sl_masked = [
            sl[(sl.index >= stdata['wind_period'][0])
               & (sl.index <= stdata['wind_period'][1])]
            for sl in stdata["geoclaw"][0]['referenced']
        ]
        overlap_times = [
            sl.index[-1] - sl.index[0] if sl.size > 0 else np.timedelta64(0, 'h')
            for sl in sl_masked
        ]
        stdata["geoclaw_exp"] = sl_masked[np.argmax(overlap_times)]
        if stdata["geoclaw_exp"].size == 0:
            stdata["geoclaw_exp"] = None

    return True


def read_waterlevel(sim, source, meta, df_annual_msl):
    df_annual_msl = df_annual_msl.loc[int(meta['date'][:4]), :]

    gaugedata = u_gauge.load_gaugedata(
        source, meta['map_id'], referenced=False, by_gsrc=False, filter_gsrc=["codec"],
        geoclaw_zos=(
            f"{ALTIMETRY_SRC}-fes_{sim.split('_')[-1]}"
            if sim.startswith("geoclaw") else None
        ),
        # resample to 10 minute averages (same as CoDEC)
        geoclaw_resample="10min",
    )

    # locations : lon/lat pairs
    locations = np.array([
        stdata[('gc_' if sim.startswith("geoclaw") else '') + 'location'][::-1]
        for stdata in gaugedata
    ])

    if locations.size == 0:
        return None, None

    if sim.startswith("geoclaw"):
        for stdata in gaugedata:
            restrict_to_wind_exposure(stdata)
        waterlevel_mm = np.array([
            0 if stdata['geoclaw_exp'] is None
            else stdata['geoclaw_exp'].values.max()
            for stdata in gaugedata
        ])
    else:
        annual_altimetry_mm = [
            1000 * df_annual_msl[stdata['filename']]
            for stdata in gaugedata
        ]
        waterlevel_mm = np.array([
            (stdata['combined'] - stdata['annual_msl'] + altimetry_mm).values.max()
            for stdata, altimetry_mm in zip(gaugedata, annual_altimetry_mm)
        ])

    waterlevel_m = waterlevel_mm / 1000

    return locations, waterlevel_m



def main():
    parser = argparse.ArgumentParser(
        description='Compute input data for bathtub tool from simulation outputs.'
    )
    parser.add_argument(
        'source',
        type=str,
        metavar="SOURCE",
        choices=['dfo', 'gfd', 'rapid'],
        help='The flood map source.')
    parser.add_argument(
        'sim',
        type=str,
        metavar="SIMULATION",
        choices=[
            "codec",
            "geoclaw-fes_no",
            "geoclaw-fes_min",
            "geoclaw-fes_mean",
            "geoclaw-fes_max",
        ],
        help='The surge height source.')
    args = parser.parse_args()
    sim = args.sim
    source = args.source

    # define a process pool (optional, for execution on a SLURM cluster)
    pool = None
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        from pathos.pools import ProcessPool as Pool
        pool = Pool(nodes=int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5")

    df_annual_msl = pd.read_hdf(u_const.GAUGES_DIR / f"annual_msl_{ALTIMETRY_SRC}.hdf5")
    df_annual_msl = df_annual_msl.set_index('years')

    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for _, row in meta.iterrows():
            st_locations, waterlevel_m = read_waterlevel(sim, source, row, df_annual_msl)
            if st_locations is None:
                # ignore floodmaps without stations (2019 and later)
                continue

            output_dir = u_const.BATHTUB_DIR / "input" / source / row['map_id']
            output_dir.mkdir(parents=True, exist_ok=True)

            ini_path = output_dir / f"inun.ini"
            if not ini_path.exists():
                print(f"Writing to {ini_path} ...")
                ini_path.write_text(
                    INI_TEMPLATE.format(source=source, map_id=row['map_id'])
                )

            nc_path = output_dir / f"{sim}.nc"
            if nc_path.exists():
                print(f"Skipping {nc_path} ...")
                continue

            write_waterlevel(nc_path, st_locations, waterlevel_m)

            if (output_dir / "dem.tif").exists():
                print(f"Skipping {output_dir / '*.tif'} ...")
                continue

            _, water_perc_data = read_raster_from_fm_meta(u_const.WATERBODY_FILE, row)
            transform, dem_data = read_raster_from_fm_meta(u_const.DEM_FILE, row)

            # the DEM data is expected to be NaN outside of land area
            dem_data[water_perc_data == 255] = np.nan

            raster_kwargs = dict(transform=transform, shape=dem_data.shape, nodata=np.nan)
            u_io.write_raster(output_dir / "occurrence.tif", water_perc_data, **raster_kwargs)
            u_io.write_raster(output_dir / "dem.tif", dem_data, **raster_kwargs)

            # write 0-valued DEM-MSL conversion file, since waterlevels are already
            # relative to the same vertical datum (the geoid)
            u_io.write_raster(output_dir / "egm.tif", 0 * dem_data, **raster_kwargs)


if __name__ == "__main__":
    main()
