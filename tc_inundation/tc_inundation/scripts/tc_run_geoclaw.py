
import os
import pathlib
import pickle
import sys
import numpy as np
import pyfes
import rasterio
import xarray as xr
from climada.hazard import Centroids, TCTracks
import climada.util.coordinates as u_coord
import climada_petals.hazard.tc_surge_geoclaw as climada_gc

INPUT_PATH = pathlib.Path("/home/gaoliang/climada/tc_inundation/input")
OUTPUT_PATH = pathlib.Path("/home/gaoliang/climada/tc_inundation/output")

ZOS_PATH = lambda name: INPUT_PATH / f"monthly_zos_{name}.nc"

DEM_PATH = INPUT_PATH / "dem"
TOPO_PATH = {
    "v1.1": DEM_PATH / "combined" / "combine_v1.1.vrt",
    "v2.1": DEM_PATH / "combined" / "combine_v2.1.vrt",
    "srtm": DEM_PATH / "srtm15plus" / "index.vrt",
}

FES_PATH = INPUT_PATH / "fes2014"


class FESReader:
    def __init__(self):
        ocean_tide_path = FES_PATH / "ocean_tide_extrapolated.ini"
        load_tide_path = FES_PATH / "load_tide.ini"
        self.short_tide = pyfes.Handler("ocean", "io", str(ocean_tide_path))
        self.radial_tide = pyfes.Handler("radial", "io", str(load_tide_path))

    def calculate(self, lons, lats, dates):
        lons, lats, dates = np.broadcast_arrays(lons, lats, dates)
        tide, lp, _ = self.short_tide.calculate(lons, lats, dates)
        load, _, _ = self.radial_tide.calculate(lons, lats, dates)
        # sum up and convert cm to m
        return (tide + lp + load) / 100.0

    def calculate_period(self, lons, lats, period, t_res_h=1):
        t_res = np.timedelta64(t_res_h, 'h')
        dates = np.arange(period[0], period[1] + t_res, t_res).astype('datetime64[us]')
        return self.calculate(lons, lats, dates)


def mean_from_zos_nc(path, period, lon, lat):
    with climada_gc._filter_xr_warnings(), xr.open_dataset(path) as ds:
        da_zos = climada_gc._nc_rename_vars(ds)["zos"]
        da_zos = da_zos.sel(time=(da_zos["time"] >= period[0]) & (da_zos["time"] <= period[1]))
        lon, lat = climada_gc._get_closest_valid_cell(da_zos, lon, lat)
        v_mean = da_zos.sel(lon=lon, lat=lat).mean().item()
    return lon, lat, v_mean


def sea_level_from_fes(zos_path, mod_zos=0, mode="mean"):
    climada_gc._sea_level_nc_info(zos_path)
    reader = FESReader()
    def sea_level_fun(bounds, period, zos_path=zos_path, reader=reader, mod_zos=mod_zos):
        centroid = (0.5 * (bounds[0] + bounds[2]), 0.5 * (bounds[1] + bounds[3]))
        period_year = (period[1] - np.timedelta64(366, "D"), period[1])

        lon, lat, nc_yrmean = mean_from_zos_nc(zos_path, period_year, *centroid)
        fes_yrmean = reader.calculate_period(lon, lat, period_year).mean()
        offset = nc_yrmean - fes_yrmean + mod_zos

        fes_value = getattr(reader.calculate_period(lon, lat, period), mode)()
        return fes_value + offset
    return sea_level_fun


def haz_to_raster(haz, path):
    """Convert point centroids to a regular grid and store as GeoTIFF file

    The raster specification is guessed from the distance between the centroids

    Parameters
    ----------
    haz : Hazard
        Object of CLIMADA's Hazard type.
    path : Path or str
        Path to a GeoTIFF file for raster output.
    """
    haz.centroids.set_lat_lon_to_meta()
    haz.centroids.meta['compress'] = 'deflate'
    assigned_idx = u_coord.match_grid_points(
        haz.centroids.lon,
        haz.centroids.lat,
        haz.centroids.meta['width'],
        haz.centroids.meta['height'],
        haz.centroids.meta['transform'],
    )
    intensity = np.zeros(
        haz.centroids.meta['height'] * haz.centroids.meta['width'],
        dtype=np.float64,
    )
    intensity[assigned_idx] = haz.intensity.toarray()
    u_coord.write_raster(path, intensity[None], haz.centroids.meta)


def centroids_from_bounds(bounds, res_as=30):
    if bounds is None:
        return None

    res_deg = res_as / (60 * 60)
    global_origin = (-180, 90)
    global_transform = rasterio.transform.from_origin(*global_origin, res_deg, res_deg)

    cens = []
    for bnd in bounds:
        centroids = Centroids()
        transform, (height, width) = u_coord.subraster_from_bounds(global_transform, bnd)
        centroids.meta = {
            'width': width,
            'height': height,
            'crs': centroids.crs,
            'transform': transform,
        }
        centroids.set_meta_to_lat_lon()
        centroids.meta = {}
        cens.append(centroids)
    return Centroids.union(*cens)


def modify_storm_intensity(ds, factor):
    ds['max_sustained_wind'] *= factor
    penv, pcen = ds.environmental_pressure, ds.central_pressure
    ds['central_pressure'] += (1 - factor) * (penv - pcen)


def main(storm_id, bounds=None, time=None, gauges=None, suffix="", dem="v2.1", zos_name="0",
         tides="no", topo_res_as=30, pool=None, mod_intensity=1, mod_zos=0):
    file_stem = f"{storm_id}{suffix}"
    out_dir = OUTPUT_PATH / file_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    path_resume = out_dir / f"{file_stem}-resume.txt"
    path_gaugedata = out_dir / f"{file_stem}-gauge_data.pickle"
    path_hdf5 = out_dir / f"{file_stem}.hdf5"
    path_tif = out_dir / f"{file_stem}.tif"

    if path_tif.exists():
        print(f"Skip already existent {out_dir} ...")
        return

    #with climada_gc._filter_xr_warnings():
    tracks = TCTracks.from_ibtracs_netcdf(storm_id=storm_id, estimate_missing=True)

    if time is not None:
        ds = tracks.data[0]
        t_mask = (time[0] <= ds.time) & (ds.time <= time[1])
        tracks.data = [ds.sel(time=t_mask)]

    if tracks.size == 0:
        sys.stderr.write(f"No valid storm data for {storm_id}, aborting ...\n")
        return

    if mod_intensity != 1:
        modify_storm_intensity(tracks.data[0], mod_intensity)

    zos_path = ZOS_PATH(zos_name)
    if tides == "no":
        sea_level_fun = climada_gc.area_sea_level_from_monthly_nc(zos_path, mod_zos=mod_zos)
    else:
        sea_level_fun = sea_level_from_fes(zos_path, mod_zos=mod_zos, mode=tides)

    haz = climada_gc.TCSurgeGeoClaw.from_tc_tracks(
        tracks,
        TOPO_PATH[dem],
        topo_res_as=topo_res_as,
        centroids=centroids_from_bounds(bounds, res_as=topo_res_as),
        gauges=gauges,
        sea_level=sea_level_fun,
        resume=path_resume,
        pool=pool,
    )

    if gauges is not None:
        with open(path_gaugedata, "wb") as fp:
            pickle.dump(haz.gauge_data, fp)
    haz.write_hdf5(path_hdf5)
    haz_to_raster(haz, path_tif)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute areas inundated by TC storm event.')
    parser.add_argument('storm_id', metavar="STORM_ID",
                        help='The IBTrACS storm ID of the storm.')
    float_tuple = lambda arg: tuple([float(x) for x in arg.strip("pPbB").split(",")])
    parser.add_argument('--bounds', type=float_tuple,
                        metavar="LON_MIN,LAT_MIN,LON_MAX,LAT_MAX", nargs="+", default=None,
                        help=('Geographical bounds. Prefix negative lon_min-values with "B" '
                              'to avoid interpretation as arguments.'))
    parser.add_argument('--time', type=np.datetime64, metavar="DATE", nargs=2, default=None,
                        help='Time window to consider (begin, end).')
    parser.add_argument('--gauges', type=float_tuple, metavar="LAT,LON", nargs="+", default=None,
                        help=('Geographical coordinates of tide gauges (lat-lon-pairs). '
                              'Prefix negative lat-values with "P" to avoid interpretation '
                              'as arguments.'))
    parser.add_argument('--zos', type=str, metavar="ZOS_NAME", default="0",
                        help='Read base sea level from this data source.')
    parser.add_argument('--dem', type=str, metavar="DEM_VERSION", default="v2.1",
                        help='Read DEM from this data source version.')
    parser.add_argument('--tides', metavar="MODE", choices=["no", "min", "mean", "max"], default="no",
                        help='Use FES2014 tide model to determine base sea level (min/mean/max).')
    parser.add_argument('--resolution', type=int, metavar="ARC_SECONDS", default=30,
                        help='Resolution of the topography in arcseconds.')
    parser.add_argument('--suffix', type=str, metavar="STR", default="",
                        help='Extra suffix for file name.')
    parser.add_argument('--mod_zos', type=float, metavar="VALUE", default=0,
                        help='Additional scalar sea level rise in meters.')
    parser.add_argument('--mod_intensity', type=float, metavar="FACTOR", default=1,
                        help='Relative change in storm intensity.')
    args = parser.parse_args()

    suffix = f"{args.suffix}-zos_{args.zos}-fes_{args.tides}"
    pool = None
    if 'SLURM_NTASKS' in os.environ and int(os.environ['SLURM_NTASKS']) > 1:
        from mpi4py.futures import MPIPoolExecutor
        pool = MPIPoolExecutor()
        # MPI is not working well at the moment, so don't overwrite existing files...
        suffix = f"{suffix}-mpi"

    main(args.storm_id, bounds=args.bounds, time=args.time, gauges=args.gauges, dem=args.dem,
         zos_name=args.zos, tides=args.tides, topo_res_as=args.resolution, suffix=suffix, pool=pool,
         mod_intensity=args.mod_intensity, mod_zos=args.mod_zos)

    if pool is not None:
        pool.shutdown()
