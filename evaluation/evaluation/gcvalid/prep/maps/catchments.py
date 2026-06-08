"""
Compute catchments from elevation data sets
"""
import argparse

import numba
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import rasterio
import rasterio.warp
from scipy.ndimage import generic_filter

import gcvalid.util.constants as u_const


def load_elevation_data(source, map_id):
    path = u_const.ELEVATION_MAPS_DIR / source / f"{map_id}.tif"
    with rasterio.open(path, "r") as src:
        data = src.read(1)
        return data, src.transform, src.crs


def generic_filter_blocked(data, win_half_size):
    result = np.zeros_like(data)
    h = win_half_size
    win_size = 2 * (2 * h + 1,)
    max_size = max(3 * h, 500)
    shape_inner = np.array(data.shape) - 2 * h
    nblocks = np.ceil(shape_inner / max_size).astype(int)
    block_sizes = np.ceil(shape_inner / nblocks).astype(int)
    for idx in np.ndindex(*nblocks):
        idx = np.array(idx)
        begins = idx * block_sizes
        ends = (idx + 1) * block_sizes + 2 * h
        pad_begin = h * (idx > 0)
        pad_end = h * (idx < nblocks - 1)
        result_index_in = tuple(
            slice(i1, i2) for i1, i2 in zip(begins + pad_begin, ends - pad_end))
        result_index_out = tuple(
            slice(h1, -h2 if h2 > 0 else None) for h1, h2 in zip(pad_begin, pad_end))
        data_index = tuple(
            slice(i1, i2) for i1, i2 in zip(begins, ends))
        result[result_index_in] = generic_filter(
            data[data_index], win_fun, size=win_size, mode="constant", cval=-1
        )[result_index_out]
    return result


@numba.njit
def win_fun(vals):
    msk = (vals >= 0)
    return ((vals[msk] >= vals[vals.size // 2]).sum() - 1) / (msk.sum() - 1)


@numba.njit
def win_fun_loop(view):
    result = np.zeros(view.shape[:2])
    h = (view.shape[-2] // 2, view.shape[-1] // 2)
    for idx in np.ndindex(*result.shape):
        N = (view[idx] >= 0).sum() - 1
        result[idx] = ((view[idx] >= view[idx + h]).sum() - 1) / N
    return result

def detect_catchments(data, res_deg, mode=0):
    # h : number of pixels for 10 km radius
    res_km = res_deg * 120
    h = int(np.ceil(10 / res_km))
    win_size = 2 * (2 * h + 1,)
    data = np.clip(data, 0, np.inf)
    if h > 100 or mode == 2:
        return win_fun_loop(sliding_window_view(
            np.pad(data, h, constant_values=-1), win_size))
    if data.size > 1000 * 1000 or mode == 1:
        return generic_filter_blocked(data, h)
    return generic_filter(data, win_fun, size=win_size, mode="constant", cval=-1)

def compute_catchment(source, meta_data):
    map_id = meta_data['map_id']
    out_path = u_const.CATCHMENTS_DIR / "windowed" / source / f"{map_id}.tif"
    if out_path.exists():
        return

    data, transform, crs = load_elevation_data(source, map_id)
    height, width = data.shape
    res_deg = meta_data['xres']

    try:
        out_data = detect_catchments(data, res_deg)
    except MemoryError:
        print(f"Skip {out_path} due to memory...")
        return

    out_kwargs = {
        "driver": "GTiff",
        "compress": "deflate",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": out_data.dtype,
        "crs": crs,
        "transform": transform,
    }
    print(f"Writing to {out_path}...")
    with rasterio.open(out_path, "w", **out_kwargs) as dst:
        dst.write(out_data, 1)

@numba.njit
def minimum_at(out, bins, weight):
    for k in range(weight.shape[0]):
        for i in range(weight.shape[1]):
            if out[k, bins[i]] > weight[k, i]:
                out[k, bins[i]] = weight[k, i]
    return out

def compute_pixel_catchment(source, meta_data):
    map_id = meta_data['map_id']
    out_path = u_const.CATCHMENTS_DIR / "by_pixel" / source / f"{map_id}.tif"
    if out_path.exists():
        return

    data_dem, transform, crs = load_elevation_data(source, map_id)
    height, width = data_dem.shape
    npixel = data_dem.size
    ocean_msk = data_dem < 0
    data_dem[ocean_msk] = 0

    path_rain = u_const.RAINFALL_MAPS_DIR / source / f"{map_id}.tif"
    with rasterio.open(path_rain, "r") as src_rain:
        data_rain = (np.arange(src_rain.height * src_rain.width)
                     .reshape(src_rain.height, src_rain.width)
                     .astype(np.float64))
        data_bins = np.full_like(data_dem, np.nan, dtype=np.float64)
        rasterio.warp.reproject(
            source=data_rain,
            destination=data_bins,
            src_transform=src_rain.transform,
            src_crs=src_rain.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=rasterio.warp.Resampling.nearest)
    data_bins = data_bins.astype(int)

    bins_uniq, bins_inverse = np.unique(data_bins.ravel(), return_inverse=True)
    data_mins_by_bin = np.full(bins_uniq.size, 10000)
    data_mins_by_bin = minimum_at(data_mins_by_bin[None, :],
                                  bins_inverse.ravel(),
                                  data_dem.ravel()[None, :])[0]
    data_mins = data_mins_by_bin[bins_inverse].reshape(data_bins.shape)
    out_data = data_dem - data_mins

    out_kwargs = {
        "driver": "GTiff",
        "compress": "deflate",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": out_data.dtype,
        "crs": crs,
        "transform": transform,
    }
    print(f"Writing to {out_path}...")
    with rasterio.open(out_path, "w", **out_kwargs) as dst:
        dst.write(out_data, 1)

def main():
    parser = argparse.ArgumentParser(description='Compute catchments from elevation data sets.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5")
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for idx, row in meta.iterrows():
            compute_catchment(source, row)
            compute_pixel_catchment(source, row)

if __name__ == "__main__":
    main()
