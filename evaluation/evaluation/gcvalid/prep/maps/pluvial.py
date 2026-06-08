"""
Create maps with pluvial flooding from catchments or DEM pixels.
"""
import argparse

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp

import gcvalid.util.constants as u_const
import gcvalid.util.io as u_io


def compute_pluvial_flood(source, meta_data, mode):
    map_id = meta_data['map_id']
    out_dir = u_const.PLUVIAL_MAPS_DIR / mode / source
    out_dir.mkdir(parents=True, exist_ok=True)
    path_out = out_dir / f"{map_id}.tif"
    path_rain = u_const.RAINFALL_MAPS_DIR / source / f"{map_id}.tif"
    path_catchment = u_const.CATCHMENTS_DIR / "windowed" / source / f"{map_id}.tif"

    data_rain = u_io.read_raster_reproject(path_rain, path_transform=path_catchment)

    data_rain[np.isnan(data_rain)] = 0
    data_rain = np.clip((data_rain - 100) / 60, 0, 5)

    if mode == "dempixels":
        path_pixel_catchment = u_const.CATCHMENTS_DIR / "by_pixel" / source / f"{map_id}.tif"
        with rasterio.open(path_pixel_catchment, "r") as src_pixel_catchment:
            data_pixel_catchment = src_pixel_catchment.read(1)
        data_flood = np.fmax(0, 2 * data_rain - data_pixel_catchment)
    else:
        with rasterio.open(path_catchment, "r") as src_catchment:
            data_catchment = src_catchment.read(1)
        data_flood = data_rain * np.clip(4 * (data_catchment - 0.5), 0, 1)

    u_io.write_raster(path_out, data_flood, path_transform=path_catchment)


def main():
    parser = argparse.ArgumentParser(description='Pluvial flooding from catchments.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5")
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for idx, row in meta.iterrows():
            compute_pluvial_flood(source, row, "dempixels")
            compute_pluvial_flood(source, row, "catchments")

if __name__ == "__main__":
    main()
