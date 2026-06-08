"""
For each gauge station, extract the annual mean sea level at the location according to the
satellite altimetry.

For stations that are outside the valid altimetry region, a larger area is averaged over.
"""
import argparse
import pickle

import climada.util.coordinates as u_coord
import numpy as np
import pandas as pd
import xarray as xr

import gcvalid.util.constants as u_const


ZOS_FILE = lambda name: u_const.WATER_DIR / f"monthly_zos_{name}.nc"
"""Path to NetCDF file with gridded altimetry data"""

PAD_DEG = 0.25
"""The amount of padding (in degrees) to add when the location of a station is not within a grid
cell with valid altimetry measurements (0.25° is the AVISO pixel size)"""


def annual_means_with_pad(ds_zos, lat, lon, pad):
    bounds = (lon - pad, lat - pad, lon + pad, lat + pad)
    zos_lon = u_coord.lon_normalize(
        ds_zos.lon.values, center=0.5 * (bounds[0] + bounds[2]))
    mask_lat = (bounds[1] <= ds_zos.lat) & (ds_zos.lat <= bounds[3])
    mask_lon = (bounds[0] <= zos_lon) & (zos_lon <= bounds[2]) & np.isfinite(ds_zos.lon)
    mask_bounds = (mask_lat & mask_lon)
    ds_zos = ds_zos.where(mask_bounds, drop=True)
    return (ds_zos.zos.mean(dim=["lat", "lon"], skipna=True)
           .groupby('time.year').mean('time'))


def annual_means(ds_zos, lat, lon):
    pad_size = 0
    nan_vals = 1
    while nan_vals > 0:
        # start with half the PAD_DEG, then increase by PAD_DEG
        pad_size += 0.5 * PAD_DEG if pad_size == 0 else PAD_DEG
        msl = annual_means_with_pad(ds_zos, lat, lon, pad_size)
        nan_vals = np.isnan(msl.values).sum()
    return msl.year.values, msl.values, pad_size


def extract_annual_means(zos_name, gaugedata, df):
    var_names = {
        'lon': ('coords', ["longitude", "lon", "x"]),
        'lat': ('coords', ["latitude", "lat", "y"]),
        'time': ('coords', ["time", "date", "datetime"]),
        'zos': ('variables', ["zos", "sla", "ssh", "adt"]),
    }
    with xr.open_dataset(ZOS_FILE(zos_name)) as ds_zos:
        for new_name, (var_type, all_names) in var_names.items():
            old_name = [c for c in getattr(ds_zos, var_type) if c.lower() in all_names][0]
            ds_zos = ds_zos.rename({old_name: new_name})
        for i_st, (stname, stloc) in enumerate(gaugedata.items()):
            print(f"\r{i_st + 1}/{len(gaugedata)}", end="")
            if df.shape[0] > 0 and df[stname].isna().sum() == 0:
                continue
            lat, lon = stloc
            if df.size > 0 and zos_name == "0":
                means = pad = 0.0
            else:
                years, means, pad = annual_means(ds_zos, lat, lon)
                if df.size == 0:
                    df['years'] = years
            df[stname].values[:] = means
            df[f'{stname}_pad'].values[:] = pad
        print("")
    return df


def get_all_stations():
    gaugedata = {}
    for gfile in u_const.GAUGES_DIR.glob("*/records/*.pickle"):
        with gfile.open("rb") as fp:
            gdata = pickle.load(fp)
        for gsrc, stations in gdata.items():
            for stdata in stations:
                if stdata['discarded'] != False and gsrc != "codec":
                    continue
                if stdata['filename'] in gaugedata:
                    continue
                gaugedata[stdata['filename']] = stdata['location']
    return gaugedata


def main():
    parser = argparse.ArgumentParser(description=(
        'Extract annual mean sea level at gauge stations according to satellite altimetry.'
    ))
    parser.add_argument('zos', type=str, metavar="SOURCE", choices=["mercator", "aviso", "0"],
                        help='The altimetry source.')
    zos = parser.parse_args().zos

    print(f"Annual mean sea levels according to {zos} for all gauge stations...")
    gaugedata = get_all_stations()
    outpath = u_const.GAUGES_DIR / f"annual_msl_{zos}.hdf5"
    columns = (
        ['years']
        + list(gaugedata.keys())
        + [f'{stname}_pad' for stname in gaugedata.keys()]
    )
    if outpath.exists():
        df = pd.read_hdf(outpath)
        # add new columns:
        new_columns = [c for c in columns if c not in df.columns]
        df = pd.concat([df, pd.DataFrame(columns=new_columns)], axis=1)
    else:
        df = pd.DataFrame(columns=columns)
    df = extract_annual_means(zos, gaugedata, df)
    print(f"Writing to {outpath} ...")
    df.to_hdf(outpath, "data")


if __name__ == "__main__":
    main()
