"""
Compute bathtub surge from CoDEC (or GeoClaw) output according to

Tiggeloven et al. (2020): Global-scale benefit–cost analysis of coastal flood adaptation
to different flood risk drivers using structural measures. Natural Hazards and Earth
System Sciences 20(4): 1025–1044.
https://nhess.copernicus.org/articles/20/1025/2020/

within flood map area at 30 arc-seconds resolution
"""
import argparse
import itertools
import os

import cartopy.io.shapereader as shapereader
import climada.util.coordinates as u_coord
from climada.util.constants import ONE_LAT_KM
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry

import gcvalid.util.constants as u_const


ATTENUATION_FACTOR = 0.5
"""Reduction in surge height (in m per km distance to coast)"""

ST_MAX_DIST_COAST_DEG = 50 * 30 / 3600
"""Only consider coastlines at most 50 km from the closest station"""

MAX_DIST_COAST_OFFSHORE_M = 5000
"""Maximum offshore distance (in m) of grid cell for consideration"""

RES_COASTLINE_DEG = 30 / 3600
"""Resolution (in degrees) of the (resampled) coastline"""

RES_GEODESIC_DEG = 15 / 3600
"""Resolution (in degrees) of the geodesic from grid cell to coastline"""


def load_surge_heights(nc_path):
    ds = xr.open_dataset(nc_path).isel(time=0)

    locations = np.stack([
        ds['station_x_coordinate'].values,
        ds['station_y_coordinate'].values,
    ], axis=1)
    surge_heights_m = ds['waterlevel'].values.copy()

    return locations, surge_heights_m


def refine_coastline(linestr):
    """Reparametrize coastline with regular spacing between nodes"""
    coords = np.array(linestr.coords)
    xp = np.zeros_like(coords[:, 0])
    xp[1:] = np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=-1))
    x = np.arange(xp[0], xp[-1], RES_COASTLINE_DEG)
    coords_fine = np.stack([
        np.interp(x, xp, coords[:, i]) for i in [0, 1]
    ], axis=1)
    return coords_fine


def compute_cell_dist_coastlines(lonlats, coastline_arr_fine):
    dists_sq = np.sum((coastline_arr_fine[None] - lonlats[:, None])**2, axis=-1)
    closest_i = np.argmin(dists_sq, axis=1)
    dists_deg = np.sqrt(np.take_along_axis(
        dists_sq, closest_i[:, None], axis=1,
    )[:, 0])
    return dists_deg, closest_i


def load_affected_coastlines(st_locations, lon, lat, pool=None):
    """Load subset of Natural Earth coastlines that are relevant for the given stations and grid cells"""
    st_points = gpd.GeoSeries(
        gpd.points_from_xy(st_locations[:, 0], st_locations[:, 1]))
    st_points_buffered = st_points.buffer(ST_MAX_DIST_COAST_DEG)
    xmin, ymin, xmax, ymax = st_points_buffered.total_bounds

    path = shapereader.natural_earth(resolution='10m', category='physical', name='coastline')
    coastlines_df = gpd.read_file(path).cx[xmin:xmax, ymin:ymax]
    coastlines_df = coastlines_df.intersection(st_points_buffered.unary_union)
    coastlines_geoms = sum([
        [geom] if isinstance(geom, shapely.geometry.LineString) else list(geom.geoms)
        for geom in coastlines_df.geometry.values
    ], [])
    coastlines_geoms = [
        linestr for linestr in coastlines_geoms if not linestr.is_empty
    ]
    assigned_stations = [
        st_points_buffered.intersects(linestr).values.nonzero()[0]
        for linestr in coastlines_geoms
    ]

    c_points = gpd.GeoSeries(gpd.points_from_xy(lon, lat))
    closest_line_to_point = np.stack([
        c_points.distance(linestr).values for linestr in coastlines_geoms
    ], axis=-1).argmin(axis=-1)
    assigned_cells = [
        (closest_line_to_point == i).nonzero()[0]
        for i in range(len(coastlines_geoms))
    ]

    refined = [refine_coastline(l) for l in coastlines_geoms]

    cell_dist_deg = []
    cell_closest_i = []
    for coastline_arr_fine, idx_cells in zip(refined, assigned_cells):
        lonlats = np.stack([lon[idx_cells], lat[idx_cells]], axis=1)
        if lonlats.size == 0:
            cell_dist_deg.append(np.array([], dtype=np.float64))
            cell_closest_i.append(np.array([], dtype=np.int64))
            continue

        chunksize = 10
        nchunks = int(np.ceil(lonlats.shape[0] / chunksize))
        lonlats_chunks = [
            lonlats[i * chunksize:(i + 1) * chunksize]
            for i in range(nchunks)
        ]
        result = (map if pool is None else pool.map)(
            compute_cell_dist_coastlines,
            lonlats_chunks,
            itertools.repeat(coastline_arr_fine, nchunks),
        )
        cell_dist_deg.append(np.concatenate([r[0] for r in result]))
        cell_closest_i.append(np.concatenate([r[1] for r in result]))

    return {
        'geoms': coastlines_geoms,
        'arrays_refined': refined,
        'assigned_stations': assigned_stations,
        'assigned_cells': assigned_cells,
        'cell_dist_deg': cell_dist_deg,
        'cell_closest_i': cell_closest_i,
    }


def weighted_coastal_surge_height(coastline, st_locations, st_surges_m):
    """To each coastline point, assign a surge height, weighted from close stations"""
    dists = np.linalg.norm(coastline[:, None] - st_locations[None], axis=-1)
    dists_inv = np.fmax(0, ST_MAX_DIST_COAST_DEG - dists)
    weights = dists_inv / dists_inv.sum(axis=-1)[:, None]
    return np.sum(weights * st_surges_m[None, :], axis=-1)


def restrict_to_connected_component(coastline, lon, lat, flood_height):
    """Set flood height to 0 outside of cells that are hydrologically connected to the sea"""
    cellcenters = gpd.GeoSeries(gpd.points_from_xy(lon, lat))
    connected_components = (
        coastline.buffer(2 * RES_GRID_DEG).union(
            cellcenters.buffer(0.6 * RES_GRID_DEG, cap_style=3).unary_union
        )
    )
    connected_components = gpd.GeoSeries(
        connected_components.geoms
        if hasattr(connected_components, "geoms")
        else [connected_components])
    ocean_components = connected_components[connected_components.intersects(coastline)]
    cell_connected_mask = cellcenters.within(ocean_components.unary_union).values

    flood_height_new = np.zeros_like(flood_height)
    flood_height_new[cell_connected_mask] = flood_height[cell_connected_mask]
    return flood_height_new


def propagate_surge(coastline_data, lon, lat, elevation_m, attenuation_reduction,
                    st_locations, st_surges_m):
    """Propagate surge heights from coastline to inland cells"""
    flood_height_m = np.zeros_like(lon)
    for i, idx_cells in enumerate(coastline_data['assigned_cells']):
        if idx_cells.size == 0:
            continue
        coastline = coastline_data['geoms'][i]
        coastline_arr_fine = coastline_data['arrays_refined'][i]
        idx_stations = coastline_data['assigned_stations'][i]
        coastline_closest_i = coastline_data['cell_closest_i'][i]

        # propagate from gauge stations to coastline
        coastal_surges_m = weighted_coastal_surge_height(
            coastline_arr_fine,
            st_locations[idx_stations],
            st_surges_m[idx_stations])

        # propagation from closest coastline point, with attenuation and elevation
        cell_surges_m = coastal_surges_m[coastline_closest_i]
        cell_attenuation_m = ATTENUATION_FACTOR * attenuation_reduction[idx_cells]
        cell_elevation_m = elevation_m[idx_cells]

        flood_height_m[idx_cells] = restrict_to_connected_component(
            coastline, lon[idx_cells], lat[idx_cells],
            np.fmax(0, cell_surges_m - cell_attenuation_m - cell_elevation_m)
        )

    return flood_height_m


def _compute_attenuation_reduction_single_cell(
        coord, coast_point, occurrence_data, occurrence_transform):
    """Compute attenuation reduction along geodesic to closest coastline point"""
    # evenly subsample geodesic from coastline point to cell
    diff = (coord - coast_point)
    dist_deg = np.linalg.norm(diff)
    n_steps = int(np.ceil(dist_deg / RES_GEODESIC_DEG))
    t = np.arange(0.5, n_steps, 1.0) / n_steps
    steps = coast_point[None] + t[:, None] * diff[None]

    # get water occurrence along geodesic
    # no attenuation in permanent water (occurrence == 100)
    occurrence = u_coord.interp_raster_data(
        occurrence_data, steps[:, 1], steps[:, 0], occurrence_transform)

    step_size_km = dist_deg * ONE_LAT_KM / n_steps
    return step_size_km * np.sum(1.0 - occurrence / 100.0)


def compute_attenuation_reduction(coastline, coastline_arr_fine, coastline_closest_i,
                                  lonlats, occurrence_path, pool=None):
    """Compute attenuation reduction along geodesic to closest coastline point"""
    coastline_arr = np.array(coastline.coords)
    all_lons = np.concatenate([coastline_arr[:, 0], lonlats[:, 0]], axis=0)
    all_lats = np.concatenate([coastline_arr[:, 1], lonlats[:, 1]], axis=0)
    bounds = (all_lons.min(), all_lats.min(), all_lons.max(), all_lats.max())
    occurrence_data, occurrence_transform = u_coord.read_raster_bounds(
        occurrence_path, bounds)
    occurrence_data = np.float64(np.clip(occurrence_data[0], 0, 100))

    return np.array(list((map if pool is None else pool.map)(
        _compute_attenuation_reduction_single_cell,
        lonlats,
        (coastline_arr_fine[i] for i in coastline_closest_i),
        itertools.repeat(occurrence_data, lonlats.shape[0]),
        itertools.repeat(occurrence_transform, lonlats.shape[0]),
    )))


def load_attenuation_reduction(input_dir, coastline_data, lon, lat, pool=None):
    """Compute (or load precomputed) attenuation reduction factors at given locations"""
    path = input_dir / "attenuation_reduction.npz"
    if path.exists():
        print(f"Load from {path} ...")
        return np.load(path)['attenuation_reduction']

    occurrence_path = input_dir / "occurrence.tif"

    attenuation_reduction = np.zeros_like(lon)
    for i, idx_cells in enumerate(coastline_data['assigned_cells']):
        if idx_cells.size == 0:
            continue
        coastline = coastline_data['geoms'][i]
        coastline_arr_fine = coastline_data['arrays_refined'][i]
        coastline_closest_i = coastline_data['cell_closest_i'][i]
        lonlats = np.stack([lon[idx_cells], lat[idx_cells]], axis=1)
        attenuation_reduction[idx_cells] = compute_attenuation_reduction(
            coastline, coastline_arr_fine, coastline_closest_i, lonlats,
            occurrence_path, pool=pool)

    print(f"Writing to {path} ...")
    np.savez_compressed(
        path, attenuation_reduction=attenuation_reduction)
    return attenuation_reduction


def compute_flood_height(input_dir, sim, pool=None):
    """Compute bathtub flood height for the specified input data"""
    # load gauge station locations and maximum surge heights (above geoid)
    st_locations, surge_heights_m = load_surge_heights(input_dir / f"{sim}.nc")
    max_surge_height_m = surge_heights_m.max()

    # infer the grid spec from the DEM file
    with rasterio.open(input_dir / "dem.tif", "r") as src:
        transform = src.transform
        elevation_m = src.read(1)
    shape = elevation_m.shape
    elevation_m = elevation_m.ravel()

    # determine potentially affected (coastal) grid cells
    lon, lat = [
        g.ravel() for g in
        u_coord.raster_to_meshgrid(transform, shape[1], shape[0])
    ]
    dist_coast_m = u_coord.dist_to_coast_nasa(lat, lon, highres=True, signed=True)
    coastal_mask = (dist_coast_m <= MAX_DIST_COAST_OFFSHORE_M) & (elevation_m < max_surge_height_m)

    # subdivide the coastline into surge affected pieces
    coastline_data = load_affected_coastlines(
        st_locations, lon[coastal_mask], lat[coastal_mask], pool=pool)

    # load precomputed attenuation reduction (from water occurrence)
    attenuation_reduction = load_attenuation_reduction(
        input_dir, coastline_data, lon[coastal_mask], lat[coastal_mask], pool=pool
    )

    flood_height_m = np.zeros_like(lon)
    flood_height_m[coastal_mask] = propagate_surge(
        coastline_data,
        lon[coastal_mask],
        lat[coastal_mask],
        elevation_m[coastal_mask],
        attenuation_reduction,
        st_locations,
        surge_heights_m,
    )

    return flood_height_m


def save_flood_height(sim, source, meta, pool=None):
    path = u_const.BATHTUB_DIR / sim / source / f"{meta['map_id']}.tif"
    if path.exists():
        return

    input_dir = u_const.BATHTUB_DIR / "aqueduct_input" / source / meta['map_id']
    if not input_dir.exists():
        print(f"No input data for {source} {meta['map_id']}. Skipping ...")
        return

    flood_height_m = compute_flood_height(input_dir, sim, pool=pool)

    print(f"Writing to {path} ...")
    dst_kwargs = {
        "driver": "GTiff",
        "compress": "deflate",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": flood_height_m.dtype,
        "crs": u_const.DEFAULT_CRS,
        "transform": transform,
    }
    with rasterio.open(path, "w", **dst_kwargs) as dst:
        dst.write(flood_height_m.reshape(shape), 1)


def main():
    parser = argparse.ArgumentParser(description=(
        'Compute bathtub floodmaps from simulation outputs.'
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
    parser.add_argument(
        'i_map',
        type=int,
        metavar="N",
        nargs='*',
        help='The flood maps to process.')
    args = parser.parse_args()

    # define a process pool (optional, for execution on a SLURM cluster)
    pool = None
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        from pathos.pools import ProcessPool as Pool
        pool = Pool(nodes=int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

    i_map = args.i_map
    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / args.source / "meta.hdf5")
    if i_map is None or len(i_map) == 0:
        i_map = np.arange(meta.shape[0])
    else:
        i_map = [i for i in i_map if i >= 0 and i < meta.shape[0]]

    if len(i_map) == 0:
        print("Nothing to do. Quitting...")
        return

    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for i in i_map:
            row = meta.iloc[i, :]
            save_flood_height(args.sim, args.source, row, pool=pool)


if __name__ == "__main__":
    main()
