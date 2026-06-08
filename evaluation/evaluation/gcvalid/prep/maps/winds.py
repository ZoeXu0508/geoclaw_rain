"""
Compute windfield of TC in flood map area (at 150 arc-seconds resolution)
"""
import argparse

from climada.hazard import TropCyclone, Centroids, TCTracks
import climada.util.coordinates as u_coord
import numpy as np
import pandas as pd
import scipy.sparse as sp
import rasterio

import gcvalid.util.constants as u_const


TIME_RES_H = 60 / 60
"""Temporal resolution of the tropical cyclone"""

SPATIAL_RES_DEG = 150 / (60 * 60)
"""Spatial resolution of the grid on which the parametric wind fields are computed"""

BOUNDS_PAD_DEG = 2 * SPATIAL_RES_DEG
"""Padding (in degrees) to apply around bounds of region of interest"""


def centroids_from_bounds(bounds):
    # cell-centered grid within padded bounds at default resolution
    global_transform = rasterio.transform.from_origin(-180, 90, SPATIAL_RES_DEG, SPATIAL_RES_DEG)
    transform, (height, width) = u_coord.subraster_from_bounds(global_transform, bounds)
    centroids = Centroids(meta=dict(
        transform=transform, width=width, height=height, crs=u_const.DEFAULT_CRS,
    ))
    centroids.set_meta_to_lat_lon()
    centroids.set_dist_coast(precomputed=True, signed=False)
    return centroids


def compute_winds(source, meta, ibtracs_data):
    map_id = meta['map_id']
    bounds = tuple(meta[['xmin', 'ymin', 'xmax', 'ymax']])
    bounds = (
        bounds[0] - BOUNDS_PAD_DEG, bounds[1] - BOUNDS_PAD_DEG,
        bounds[2] + BOUNDS_PAD_DEG, bounds[3] + BOUNDS_PAD_DEG,
    )
    outfile = u_const.WINDS_DIR / source / f"{map_id}.tif"

    if outfile.exists():
        return

    tracks = ibtracs_data.subset({'sid': map_id[:-2]})

    # restrict the track not to exceed the end of the day of the flood map
    # so that we get only the wind field up to (and including) that day
    fm_date = meta['date'] + np.timedelta64(24, 'h')
    track_ds = tracks.data[0]
    track_ds = track_ds.sel(time=track_ds.time <= fm_date)
    tracks.data = [track_ds]

    centroids = centroids_from_bounds(bounds)

    if track_ds.time.size < 2:
        haz = TropCyclone()
        haz.centroids = centroids
        haz.intensity = sp.csr_matrix(
            ([], [], [0, 0]), shape=(1, centroids.size))
    else:
        haz = TropCyclone.from_tracks(tracks, centroids)

    # 50 km is hardcoded in the CLIMADA GeoClaw setup anyways, so truncate:
    # Note: This is imposing an implicit 50 km in-land mask for the coastal compare masks!
    mask = (haz.centroids.dist_coast <= 50000).astype(np.float64)
    haz.intensity = haz.intensity.dot(sp.diags(mask))
    haz.centroids.meta['compress'] = 'deflate'

    print(f"Writing to {outfile} ...")
    haz.write_raster(outfile, intensity=True)


def main():
    parser = argparse.ArgumentParser(description='Compute windfield of TC in flood map area.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5")
    meta['date'] = pd.to_datetime(meta.date)
    ibtracs_data = TCTracks.from_netcdf(u_const.TRACKS_DIR / source)
    ibtracs_data.equal_timestep(time_step_h=TIME_RES_H)
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for idx, row in meta.iterrows():
            compute_winds(source, row, ibtracs_data)


if __name__ == "__main__":
    main()
