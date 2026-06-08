"""
For each gauge station, find a location that is not above sea level according to the topography
evaluated at the 30 arc-second reference grid
"""
import argparse
import pathlib
import pickle

import climada.util.coordinates as u_coord
import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage

import gcvalid.util.constants as u_const


def closest_offshore_point(lat, lon):
    # within a 4 km radius around the given point, find the closest pixel in the DEM that is below
    # sea level, and for which all neighboring points are also below sea level
    res = 30 / (60 * 60)
    d_pad = 2 * res
    pad = 0
    while True:
        pad += d_pad
        bounds = (lon - pad, lat - pad, lon + pad, lat + pad)
        # Make sure that we read the DEM with the same interpolation method ("average")
        # and on the exact same grid (30as) as is done in the GeoClaw module:
        zvalues, transform = u_coord.read_raster_bounds(
            u_const.DEM_FILE,
            bounds,
            res=res,
            bands=[1],
            resampling="average",
            global_origin=(-180, 90),
        )

        # use maximum_filter to include 8 neighboring points
        zvalues = scipy.ndimage.maximum_filter(zvalues[0], size=3)
        mask = (zvalues <= -0.1)

        # never select points on the boundary (because of missing neighbors)
        bdry_mask = np.ones_like(mask)
        bdry_mask[1:-1, 1:-1] = False
        mask[bdry_mask] = False

        if mask.sum() == 0:
            # when there is no such point, increase search radius
            continue

        height, width = zvalues.shape
        longrid, latgrid = u_coord.raster_to_meshgrid(transform, width, height)
        longrid = longrid[mask]
        latgrid = latgrid[mask]
        dists = u_coord.dist_approx(
            latgrid[None], longrid[None],
            np.array([[lat]]), np.array([[lon]]),
            method="geosphere")[0, :, 0]
        i = dists.argmin()
        return latgrid[i], longrid[i]


def get_gc_location(stdata, locations, gc_locations):
    stname = stdata['filename']
    if stname not in locations:
        locations[stname] = stdata['location']
    if 'gc_location' in stdata:
        gc_locations[stname] = stdata['gc_location']
        return False
    if stname in gc_locations:
        stdata['gc_location'] = gc_locations[stname]
    else:
        stdata['gc_location'] = closest_offshore_point(*stdata['location'])
        gc_locations[stname] = stdata['gc_location']
    return True


def main():
    parser = argparse.ArgumentParser(description=(
        'Find nearby location for gauge stations that is offshore according to the DEM.'
    ))
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    gc_locations = {}
    locations = {}
    all_files = sorted([str(p) for p in u_const.GAUGES_DIR.glob(f"{source}/records/*.pickle")])
    for ifile, gfile in enumerate(all_files):
        print(f"Processing file {ifile:3d}/{len(all_files)} ...", end="\r", flush=True)
        gfile = pathlib.Path(gfile)
        with gfile.open("rb") as fp:
            gdata = pickle.load(fp)
        results = []
        for gsrc, stations in gdata.items():
            results.extend([
                get_gc_location(stdata, locations, gc_locations)
                for stdata in stations
                if stdata['discarded'] == False or gsrc == "codec"
            ])
        if not any(results):
            continue
        print(f"Writing to {gfile} ...")
        with gfile.open("wb") as fp:
            pickle.dump(gdata, fp)
    if len(locations) == 0 or len(gc_locations) == 0:
        print("No gauge data found for any of the flood maps from this source!")
        return
    df = pd.concat([pd.DataFrame(locations), pd.DataFrame(gc_locations)]).transpose()
    df.columns = ['lat', 'lon', 'gc_lat', 'gc_lon']
    df['dist_km'] = u_coord.dist_approx(
        df['lat'].values[:, None], df['lon'].values[:, None],
        df['gc_lat'].values[:, None], df['gc_lon'].values[:, None],
        method="geosphere")[:, 0, 0]
    print(df[df['dist_km'] > 2])


if __name__ == "__main__":
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        main()
