"""
Postprocess output data for aqueduct bathtub tool:

https://github.com/Deltares/aqueduct-coastal-flooding/blob/py38/

Convert from NetCDF to GeoTIFF
"""
import argparse

import numpy as np
import rasterio
import xarray as xr

import gcvalid.util.constants as u_const


def convert_nc_to_tif(inpath):
    outpath = inpath.parent / inpath.name.replace(".nc", ".tif")

    da = xr.open_dataset(inpath).isel(time=0)['inun']
    xmin = da.lon.values.min()
    dx = da.lon.values[1] - da.lon.values[0]
    ymax = da.lat.values.max()
    dy = da.lat.values[1] - da.lat.values[0]
    transform = rasterio.Affine(dx, 0, xmin - 0.5 * dx,
                                0, -dy, ymax + 0.5 * dy)
    data = np.flip(da.values, axis=0)

    print(f"Writing to {outpath} ...")
    dst_kwargs = {
        "driver": "GTiff",
        "compress": "deflate",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype,
        "crs": u_const.DEFAULT_CRS,
        "transform": transform,
    }
    with rasterio.open(outpath, "w", **dst_kwargs) as dst:
        dst.write(data, 1)


def main():
    parser = argparse.ArgumentParser(description=(
        'Convert output data from aqueduct bathtub tool to GeoTIFF.'
    ))
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

    output_dir = u_const.BATHTUB_DIR / args.sim / args.source / "aqueduct_output"
    for path in output_dir.glob("*/inun.nc"):
        convert_nc_to_tif(path)


if __name__ == "__main__":
    main()
