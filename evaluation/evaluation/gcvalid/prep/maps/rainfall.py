"""
Get 21-day (from reference date backwards) rainfall in flood map area

- daily MAXIMUM rainfall within flood map area (supposed to be independent of area size)
- rainfall is only available until end of 2018 from WFDE5
- compare WFDE5 and ERA5 and let's see how it differs...
"""
import argparse

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import xarray as xr

from gcvalid.util import dt64_to_dmy
import gcvalid.util.constants as u_const
import gcvalid.util.io as u_io


REF_PERIOD = (2001, 2018)


def rainfall_for_map(source, meta):
    map_id = meta['map_id']
    bounds = tuple(meta[['xmin', 'ymin', 'xmax', 'ymax']])

    date = np.datetime64(meta['date'])
    day, month, year = dt64_to_dmy(date)
    days = [date - np.timedelta64(i, "D") for i in range(21)]

    out_path = u_const.RAINFALL_MAPS_DIR / source / f"{map_id}.csv"
    if not out_path.exists():
        days += [date - np.timedelta64(i, "D") - np.timedelta64(int(365.25 * (year - yr)), "D")
                 for yr in range(REF_PERIOD[0], REF_PERIOD[1] + 1) for i in range(21)]
        df_data = {"date": days}
        source_names = {
            "gpcc": "GPCC",
            "wfde5": "WFDE5",
            "era5l": "ERA5-Land",
            "era5_combined": "ERA5-combined",
        }
        for source, name in source_names:
            df_data[name] = [u_io.read_daily_rainfall(source, d, bounds) for d in days]

        print(f"Writing to {out_path}...")
        pd.DataFrame(df_data).to_csv(out_path, index=False)

    out_path = u_const.RAINFALL_MAPS_DIR / source / f"{map_id}.tif"
    if not out_path.exists():
        rainfall_era5c = [
            u_io.read_daily_rainfall("era5_combined", d, bounds, spatial_max=False)
            for d in days
        ]
        transform = rainfall_era5c[0][1]
        data = np.sum([r[0] for r in rainfall_era5c], axis=0)
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
    parser = argparse.ArgumentParser(description='Get rainfall in flood map area.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5")
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for idx, row in meta.iterrows():
            rainfall_for_map(source, row)


if __name__ == "__main__":
    main()
