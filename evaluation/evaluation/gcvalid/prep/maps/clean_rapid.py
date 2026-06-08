"""
Split RAPID flood maps and reduce to values 0/1/2
"""
import os

import numpy as np
import pandas as pd
import rasterio
import rasterio.merge
import rasterio.warp

import gcvalid.util.constants as u_const


RAPID_DATA_DIR = u_const.FLOODMAPS_DIR / "rapid"

RAPID_INDEX_DIR = u_const.INDEX_DIR / "rapid"

SUBAREAS = {
    "2016-10-02-2016-11-02-4601": [  # 2016273N13300
        (0.35, 0, 0.9, 0.5),
        (0, 0.5, 0.5, 1.0),
    ],
    "2017-08-25-2017-09-13-3424": [  # 2017228N14314
        (0, 0, 1, 1),
    ],
    "2017-09-06-2017-11-11-5796": [  # 2017242N16333
        (0, 0, 1, 0.5),
        (0, 0.5, 1, 1),
    ],
    "2018-09-11-2018-10-22-1625": [  # 2018242N13343
        (0, 0, 1, 1),
    ],
    "2018-10-10-2018-10-14-4255": [  # 2018280N18273
        (0, 0, 0.5, 1),
        (0.5, 0, 1, 1),
    ],
    "2019-07-10-2019-07-13-132": [  # 2019192N29274
        (0, 0, 1, 1),
    ],
}
"""Subdivisions of the original flood maps.

The coordinates (left, top, right, bottom) are relative coordinates ranging from 0.0 to 1.0
from top to bottom (!) and from left to right
"""


def read_subarea(rapid_id, shape, transform):
    flood = np.full(shape, 255, dtype=np.uint8)
    for path in (RAPID_DATA_DIR / "raw" / rapid_id).glob("*/*/flood_*.tif"):
        with rasterio.open(path, "r") as src:
            data = np.full(flood.shape, 255, dtype=np.uint8)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=src.crs,
                resampling=rasterio.warp.Resampling.average)
            mask = ((0 < data) & (data < 255)) | ((data == 0) & (flood != 1))
            flood[mask] = np.ceil(data[mask]).astype(int)
    for path in (RAPID_DATA_DIR / "raw" / rapid_id).glob("*/*/non_flood_*.tif"):
        with rasterio.open(path, "r") as src:
            data = np.full(flood.shape, 255, dtype=np.uint8)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=src.crs,
                resampling=rasterio.warp.Resampling.average)
            flood[(0 < data) & (data < 255)] = 2
    return flood


def extract_subarea(rapid_id, shape, transform, out_path):
    data = read_subarea(rapid_id, shape, transform)
    kwargs = {
        "driver": "GTiff",
        "compress": "deflate",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": data.dtype,
        "nodata": 255,
        "crs": u_const.DEFAULT_CRS,
        "transform": transform,
    }
    print(f"Writing to {out_path}...")
    with rasterio.open(out_path, "w", **kwargs) as dst:
        dst.write(data, 1)


def get_full_rapid_area(rapid_id):
    in_paths = (RAPID_DATA_DIR / "raw" / rapid_id).glob("*/*/*flood_*.tif")
    flood, transform = rasterio.merge.merge(
        list(in_paths), indexes=[1], method="last", res=3 / (60 * 60))
    return flood.shape[1:], transform


def get_subareas(rapid_id):
    shape, transform = get_full_rapid_area(rapid_id)
    xres, _, xmin, _, yres, ymax = transform[:6]
    assert xres > 0
    assert yres < 0
    for sub_xmin, sub_ymin, sub_xmax, sub_ymax in SUBAREAS[rapid_id]:
        imin, imax = np.floor(sub_ymin * shape[0]), np.ceil(sub_ymax * shape[0])
        jmin, jmax = np.floor(sub_xmin * shape[1]), np.ceil(sub_xmax * shape[1])
        sub_transform = rasterio.Affine(
            xres, 0, xmin + jmin * xres,
            0, yres, ymax + imin * yres)
        sub_shape = (int(imax - imin), int(jmax - jmin))
        yield sub_shape, sub_transform


def process_rapid_map(meta):
    rapid_id, ibtracs_id = meta[['rapid_id', 'ibtracs_id']]
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for i, (shape, transform) in enumerate(get_subareas(rapid_id)):
            out_path = RAPID_DATA_DIR / "clean_by_sid" / f"{ibtracs_id}-{i}.tif"
            if out_path.exists():
                continue
            extract_subarea(rapid_id, shape, transform, out_path)


def main():
    df = pd.read_csv(RAPID_INDEX_DIR / "linked_ids.csv")

    # define a process pool (optional, for execution on a SLURM cluster)
    pool = None
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        from pathos.pools import ProcessPool as Pool
        pool = Pool(nodes=int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

    if pool is None:
        for _, row in df.iterrows():
            process_rapid_map(row)
    else:
        pool.map(process_rapid_map, [row for _, row in df.iterrows()])

if __name__ == "__main__":
    main()
