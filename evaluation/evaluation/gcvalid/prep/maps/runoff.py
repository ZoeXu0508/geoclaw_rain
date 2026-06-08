"""
Get 21-day (from reference date backwards) surface runoff in flood map area
"""
import argparse

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import xarray as xr

import gcvalid.util.constants as u_const
import gcvalid.util.io as u_io


def runoff_for_map(source, meta):
    map_id = meta['map_id']
    bounds = tuple(meta[['xmin', 'ymin', 'xmax', 'ymax']])

    date = np.datetime64(meta['date'])
    days = [date - np.timedelta64(i, "D") for i in range(21)]

    out_path = u_const.RUNOFF_MAPS_DIR / source / f"{map_id}.tif"
    if not out_path.exists():
        runoff_era5c = [
            u_io.read_daily_rainfall("era5_combined", d, bounds, spatial_max=False, sro=True)
            for d in days
        ]
        transform = runoff_era5c[0][1]
        data = np.sum([r[0] for r in runoff_era5c], axis=0)
        kwargs = {
            "driver": "GTiff",
            "compress": "deflate",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": np.float32,
            "crs": u_const.DEFAULT_CRS,
            "transform": transform,
        }
        print(f"Writing to {out_path}...")
        with rasterio.open(out_path, "w", **kwargs) as dst:
            dst.write(data, 1)


def main():
    parser = argparse.ArgumentParser(description='Get surface runoff in flood map areas.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5")
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for idx, row in meta.iterrows():
            runoff_for_map(source, row)


if __name__ == "__main__":
    main()
