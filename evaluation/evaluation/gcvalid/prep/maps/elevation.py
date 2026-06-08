"""
Get elevation within flood map's area according to mixed DEM data set
"""
import argparse

import climada.util.coordinates as u_coord
import numpy as np
import pandas as pd
import rasterio

import gcvalid.util.constants as u_const
import gcvalid.util.io as u_io


RES_GRID_DEG = 30 / 3600
"""Resolution (in degrees) of the output raster"""


def grid_from_bounds(bounds):
    global_transform = rasterio.transform.from_origin(-180, 90, RES_GRID_DEG, RES_GRID_DEG)
    return u_coord.subraster_from_bounds(global_transform, bounds)


def read_raster_from_fm_meta(path, meta_data, resampling="average"):
    pad = 5 * RES_GRID_DEG
    bounds = tuple(meta_data[['xmin', 'ymin', 'xmax', 'ymax']])
    bounds = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
    transform, shape = grid_from_bounds(bounds)
    data = u_io.read_raster_reproject(
        path,
        resampling=resampling,
        transform=transform,
        shape=shape,
        dtype=np.float64,
    )
    return transform, data


def write_elevation(source, meta_data):
    map_id = meta_data['map_id']
    out_path = u_const.ELEVATION_MAPS_DIR / source / f"{map_id}.tif"
    if out_path.exists():
        return

    transform, data = read_raster_from_fm_meta(u_const.DEM_FILE, meta_data)

    print(f"Writing to {out_path}...")
    u_io.write_raster(out_path, data, transform=transform, shape=data.shape, nodata=np.nan)


def main():
    parser = argparse.ArgumentParser(description='Get elevation within flood map area.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5")
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for _, row in meta.iterrows():
            write_elevation(source, row)

if __name__ == "__main__":
    main()
