"""
Compute bathtub surge (according to CLIMADA) of TC in flood map
area (at 30 arc-seconds resolution)
"""
import argparse

from climada.hazard import TropCyclone, Centroids, TCTracks
import climada.util.coordinates as u_coord
from climada_petals.hazard import TCSurgeBathtub
import numpy as np
import pandas as pd
import scipy.sparse as sp
import rasterio

import gcvalid.util.constants as u_const


RES_GRID_DEG = 30 / 3600
"""Resolution (in degrees) of the output raster"""


def centroids_from_bounds(bounds):
    global_origin = (-180, 90)
    global_transform = rasterio.transform.from_origin(*global_origin, RES_GRID_DEG, RES_GRID_DEG)

    centroids = Centroids()
    transform, (height, width) = u_coord.subraster_from_bounds(global_transform, bounds)
    centroids.meta = {
        'width': width,
        'height': height,
        'crs': centroids.crs,
        'transform': transform,
    }
    centroids.set_meta_to_lat_lon()
    centroids.meta = {}
    centroids.set_dist_coast(precomputed=True)
    return centroids


def compute_surge(source, meta, ibtracs_data):
    map_id = meta['map_id']
    pad = 2 * RES_GRID_DEG
    bounds = tuple(meta[['xmin', 'ymin', 'xmax', 'ymax']])
    bounds = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)

    raster_file = u_const.BATHTUB_DIR / "climada" / source / f"{map_id}.tif"
    if raster_file.exists():
        return

    tracks = ibtracs_data.subset({'sid': map_id[:-2]})
    winds = TropCyclone.from_tracks(tracks, centroids_from_bounds(bounds))
    dem_path = u_const.ELEVATION_MAPS_DIR / source / f"{map_id}.tif"
    haz = TCSurgeBathtub.from_tc_winds(winds, dem_path)

    haz.centroids.set_lat_lon_to_meta()
    haz.centroids.meta['compress'] = 'deflate'
    print(f"Writing to {raster_file}")
    haz.write_raster(raster_file, intensity=True)


def main():
    parser = argparse.ArgumentParser(description="Compute CLIMADA's bathtub surge.")
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5")
    ibtracs_data = TCTracks.from_netcdf(u_const.TRACKS_DIR / source)
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for idx, row in meta.iterrows():
            compute_surge(source, row, ibtracs_data)


if __name__ == "__main__":
    main()
