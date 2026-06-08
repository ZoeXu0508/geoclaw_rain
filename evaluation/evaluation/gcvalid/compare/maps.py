"""
Compare flood maps with the respective GeoClaw model runs
"""
import argparse

import climada.util.coordinates as u_coord
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.crs
import rasterio.warp

import gcvalid.util.constants as u_const
import gcvalid.util.io as u_io


DEF_RESOLUTION = 30 / (60 * 60)

DEF_CRS = rasterio.crs.CRS.from_string(u_const.DEFAULT_CRS)

ONE_LAT_KM = 111.12
"""Mean one latitude (in degrees) to km"""

PLUVIAL_MODES = ["without"] + [
    f"{f}_{pl}" for f in ["w", "o"] for pl in ["isimip2a", "dempixels", "catchments"]
] + [
    f"{f}_isimip3a{p}" for f in ["w", "o"] for p in ["noprot", "2yprot", "flopros"]
] + [
    f"bt_{bt}" for bt in ["climada", "codec", "aq_codec"]
] + [
    f"bt_{aq}geoclaw-fes_{f}" for aq in ["", "aq_"] for f in ["no", "min", "mean", "max"]
] + [
    f"fm_{fm}" for fm in ["dfo", "gfd", "rapid"]
]

DEM_COMMENT = {
    "2003262N17254-0": "CoastalDEM: rectangular artifact (value=0) in waterbody",  # still present in v2.1
    "2004061S12072-0": "CoastalDEM: rectangular artifact (value=0) in waterbody",  # still present in v2.1
    "2006237N13298-0": "All areas: polygonal artifact close to Myrtle Beach",  # fixed in v2.1
    "2016273N13300-1": "All areas: polygonal artifact close to Myrtle Beach",  # fixed in v2.1
}

DFO_COMBINE_MAPS = {
    # the first is overwritten by the (not-NaN) data in the subsequently listed data sets
    "2005236N23285": ["3", "4", "2", "0"],
    "2005275N19274": ["3", "2", "1"],
}
"""Combine maps from DFO that are contained within each other to avoid double-counting."""


def write_compare_tif(path, data, transform):
    kwargs = {
        "driver": "GTiff",
        "compress": "deflate",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": np.uint8,
        "crs": DEF_CRS,
        "transform": transform,
    }
    print(f"Writing to {path} ...")
    with rasterio.open(path, "w", **kwargs) as dst:
        dst.write(data, 1)


def gen_compare_data(waterbody_data, dem_data, wind_data, fm_data, haz_data, model_thresh=0):
    # permanent water body in flood map enlarges (but doesn't shrink) external waterbody data
    # missing observations (255) are considered permanent water body in the comparison
    waterbody_data[(fm_data == 2) | (fm_data == 255)] = 100

    # it happens that one row/col of pixels at the boundary has no wind data due to reprojection
    wind_data[np.isnan(wind_data)] = 0.0

    # 0 = coastal no flooding,
    # 1-11 = observational,
    # 12-22 = geoclaw 0-5 m inundation height in 0.5m steps,
    # 23-33 = both,
    # 255 = cutoff no flooding (or water occurrence >5%),
    # 254 = cutoff obs,
    # 253 = cutoff geoclaw,
    # 252 = cutoff both,
    # cutoff: height >10m or height <-10m or wind == 0
    # Note: The wind cutoff imposes an implicit 50 km in-land threshold!
    data = 1 * (fm_data > 0).astype(np.uint8) + 2 * (haz_data > model_thresh).astype(np.uint8)
    data[waterbody_data > 5] = 255
    data[np.isnan(haz_data)] = 255
    dem_mask = ((np.abs(dem_data) > 10) | (wind_data <= 0.0)) & (data != 255)
    data[dem_mask] = 255 - data[dem_mask]

    for i in range(3):
        data[data == i + 1] += i * 10
    geoclaw_msk = (data == 12) | (data == 23)
    data[geoclaw_msk] += (haz_data[geoclaw_msk] * 2).clip(0, 10).astype(np.uint8)

    return data


def load_fm_data(source, map_id, source_mask):
    ibtracs_id, map_no = map_id.split("-")
    map_nos = (
        DFO_COMBINE_MAPS[ibtracs_id] if (
            source == "dfo" and map_no in DFO_COMBINE_MAPS.get(ibtracs_id, [])
        ) else [map_no]
    )
    paths = [
        u_const.FLOODMAPS_DIR / source / "clean_by_sid" / f"{ibtracs_id}-{n}.tif"
        for n in map_nos
    ]

    dst_bounds = (np.inf, np.inf, -np.inf, -np.inf)
    src_spec = []
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for p in paths:
            with rasterio.open(p, "r") as src:
                b = tuple(src.bounds)
                dst_bounds = (
                    min(dst_bounds[0], b[0]), min(dst_bounds[1], b[1]),
                    max(dst_bounds[2], b[2]), max(dst_bounds[3], b[3]),
                )
                d = src.read(1)
                # missing observations (255) are considered as permanent water body (2)
                d[d == 255] = 2
                src_spec.append(
                    (d.astype(np.float64), src.crs, src.transform)
                )

    dst_data = None
    for src_d, src_crs, src_transform in src_spec:
        d, dst_transform = u_coord.align_raster_data(
            src_d, src_crs, src_transform, src_nodata=2,
            dst_crs=DEF_CRS, dst_resolution=DEF_RESOLUTION, dst_bounds=dst_bounds,
            global_origin=(-180, 90), resampling=rasterio.warp.Resampling.average,
        )
        d = np.ceil(d).astype(int)
        if dst_data is None:
            dst_data = d
        else:
            fin_mask = (d != 2)
            dst_data[fin_mask] = d[fin_mask]
    if map_no != map_nos[0]:
        dst_data[:] = 2

    if source_mask is not None:
        dst_shape = dst_data.shape
        ibtracs_id = map_id.split("-")[0]
        paths = fm_paths_from_ibtracs_id(source_mask, ibtracs_id)
        for p in paths:
            msk_data = reproject_raster_file(p, dst_shape, dst_transform, "average", True)
            dst_data[np.isnan(msk_data)] = 2
        if len(paths) == 0:
            # assume all-NaNs if there is no mask data for this storm
            dst_data[:] = 2

    return dst_data, dst_bounds, dst_transform


def geoclaw_output_tif(source, ibtracs_id, zos):
    return u_const.GEOCLAW_DIR / source / "results" / f"{ibtracs_id}_{source}-zos_{zos}.tif"


def fm_paths_from_ibtracs_id(source, ibtracs_id):
    return sorted(
        (u_const.FLOODMAPS_DIR / source / "clean_by_sid").glob(f"{ibtracs_id}-*.tif"),
        key=lambda p: p.stem,
    )


def reproject_raster_file(path, dst_shape, dst_transform, resampling, is_fm_data):
    p_data = np.full(dst_shape, np.nan)
    with rasterio.open(path, "r") as src:
        if is_fm_data:
            src_nodata = 2
            src_data = src.read(1)
            src_data[src_data == 255] = 2
            src_data = src_data.astype(np.float64)
        else:
            src_nodata = src.nodata
            src_data = rasterio.band(src, 1)
        rasterio.warp.reproject(
            source=src_data,
            destination=p_data,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=DEF_CRS,
            dst_nodata=np.nan,
            resampling=getattr(rasterio.warp.Resampling, resampling),
        )
    if is_fm_data:
        p_data = np.ceil(p_data)
    return p_data


def load_raster_data(source, map_id, zos, fm_bounds, fm_shape, fm_transform, pluvial, extra=True):
    standalone_pluv = any(pluvial.startswith(s) for s in ["o_", "bt_", "fm_"])

    ibtracs_id, map_no = map_id.split("-")
    combine_map_ids = (
        [f"{ibtracs_id}-{n}" for n in DFO_COMBINE_MAPS[ibtracs_id]] if (
            source == "dfo" and map_no in DFO_COMBINE_MAPS.get(ibtracs_id, [])
        ) else [map_id]
    )

    with rasterio.Env(VRT_SHARED_SOURCE=False):
        data = []
        paths = [
            # The resampling method ("bilinear"/"average") is really important for cases where the
            # resolution of source and destination are very different, most notably for the water
            # occurrence (1as) and the DEM (3as), but also for winds (150as).
            # If the resolution agrees, "bilinear" and "average" are identical (tested with
            # rasterio on 2023-01-30), which applies for GeoClaw and pluvial data.
            ([geoclaw_output_tif(source, map_id[:-2], zos)], "bilinear"),
        ]
        if extra:
            paths.append(([u_const.WATERBODY_FILE], "average"))
            # The elevation is already on the right grid. Hence, the interpolation
            # method (e.g., "average" or "bilinear") does not really matter.
            paths.append(([
                u_const.ELEVATION_MAPS_DIR / source / f"{m}.tif"
                for m in combine_map_ids
            ], "average"))
            paths.append(([
                u_const.WINDS_DIR / source / f"{m}.tif"
                for m in combine_map_ids
            ], "bilinear"))
        if pluvial.startswith("fm_"):
            ibtracs_id = map_id.split("-")[0]
            fm_source = pluvial.split("_")[1]
            paths.append(
                (fm_paths_from_ibtracs_id(fm_source, ibtracs_id), "average")
            )
        elif pluvial != "without":
            paths.append(([
                u_const.BATHTUB_DIR / pluvial[6:] / source / "aqueduct_output" / m / "inun.tif"
                if pluvial.startswith("bt_aq_") else
                u_const.BATHTUB_DIR / pluvial[3:] / source / f"{m}.tif"
                if pluvial.startswith("bt_") else
                u_const.PLUVIAL_MAPS_DIR / pluvial[2:] / source / f"{m}.tif"
                for m in combine_map_ids
            ], "bilinear"))
        for i, (path, resampling) in enumerate(paths):
            if i == 0 and standalone_pluv:
                data.append(None)
                continue
            is_fm_data = (i == len(paths) - 1) and pluvial.startswith("fm_")

            path = [p for p in path if p.exists()]
            if len(path) == 0:
                data.append(None)
                continue

            d = np.full(fm_shape, np.nan)
            for p in path:
                p_data = reproject_raster_file(p, fm_shape, fm_transform, resampling, is_fm_data)
                nan_mask = np.isnan(d)
                d[nan_mask] = p_data[nan_mask]
            data.append(d)
        if pluvial != "without":
            haz_data = data[0]
            pluv_data = data[-1]
            extra_data = data[1:-1]
            if pluv_data is None:
                print(
                    f"No pluvial data for {source} {map_id},"
                    f" assuming {'NaNs' if is_fm_data else 'zeros'} ..."
                )
                pluv_data = np.full(fm_shape, np.nan if is_fm_data else 0.0)
            haz_data = (
                pluv_data if standalone_pluv else np.fmax(haz_data, pluv_data)
            )
            data = [haz_data] + extra_data
    return data


def generate_compare_tifs(source_m, zos, pluvial, model_thresh):
    print(f"Generating compare TIFFs for {source_m} "
          f"(zos={zos}, pluvial={pluvial}, model_thresh={model_thresh})...")
    compare_dir = u_const.COMPARE_DIR / source_m / pluvial / zos

    source, source_mask = (
        source_m.split("_") if "_" in source_m else (source_m, None)
    )

    map_ids = sorted([
        f.stem for f in (u_const.FLOODMAPS_DIR / source / "clean_by_sid").glob("*.tif")
    ])
    for i_map_id, map_id in enumerate(map_ids):
        print(f"{i_map_id}/{len(map_ids)}", end="\r", flush=True)

        outfile = compare_dir / f"{map_id}-thresh_{model_thresh:.1f}.tif"
        if outfile.exists():
            continue

        if not geoclaw_output_tif(source, map_id[:-2], zos).exists():
            print(f"GeoClaw data for {source} ({map_id}, {zos}) doesn't exist! Skipping...")
            continue

        fm_data, fm_bounds, compare_transform = load_fm_data(source, map_id, source_mask)
        haz_data, waterbody_data, dem_data, wind_data = load_raster_data(
            source, map_id, zos, fm_bounds, fm_data.shape, compare_transform, pluvial)
        compare_data = gen_compare_data(
            waterbody_data, dem_data, wind_data, fm_data, haz_data, model_thresh=model_thresh)
        write_compare_tif(outfile, compare_data, compare_transform)
    print(f"{len(map_ids)}/{len(map_ids)}")


def compute_compare_stats(source_m, zos, pluvial, model_thresh):
    print("Computing statistics from previously generated TIFFs...")
    compare_dir = u_const.COMPARE_DIR / source_m / pluvial / zos

    source, source_mask = (
        source_m.split("_") if "_" in source_m else (source_m, None)
    )

    rows = []
    for path in compare_dir.glob(f"*-thresh_{model_thresh:.1f}.tif"):
        with rasterio.open(path, "r") as src:
            data = src.read(1)
            transform = src.transform
        xgrid, ygrid = u_coord.raster_to_meshgrid(transform, data.shape[1], data.shape[0])
        coastal_colored_msk = (data > 0) & (data < 50)
        dlon, dlat = np.abs(transform[0]), np.abs(transform[4])
        if coastal_colored_msk.sum() > 0:
            lon_min = xgrid[coastal_colored_msk].min() - 0.5 * dlon
            lon_max = xgrid[coastal_colored_msk].max() + 0.5 * dlon
            lat_min = ygrid[coastal_colored_msk].min() - 0.5 * dlat
            lat_max = ygrid[coastal_colored_msk].max() + 0.5 * dlat
        else:
            lon_min = lon_max = xgrid.mean()
            lat_min = lat_max = ygrid.mean()
        rows.append([
            path.stem.split("-thresh_")[0],
            0.5 * (np.abs(transform[0]) + np.abs(transform[4])) * (60 * 60),
            dlon, dlat, lon_min, lon_max, lat_min, lat_max,
            (data > 50).sum(),
            (data == 254).sum() + (data == 252).sum(),
            (data == 253).sum() + (data == 252).sum(),
            (data == 0).sum() + coastal_colored_msk.sum(),
            (data == 1).sum(),
        ] + [(data == 12 + i).sum() for i in range(11)
        ] + [(data == 23 + i).sum() for i in range(11)])

    df = pd.DataFrame(
        data=rows,
        columns=["map_id", "res_as", "dlon", "dlat", "lon_min", "lon_max", "lat_min", "lat_max",
                 "cutoff_total", "cutoff_fm_total", "cutoff_gc_total",
                 "coastal_total", "coastal_fm_total"
                ] + [f"coastal_gc{i}_total" for i in range(11)
                ] + [f"coastal_both{i}_total" for i in range(11)])

    for n in ['both', 'gc']:
        df[f'coastal_{n}_total'] = df[[f"coastal_{n}{i}_total" for i in range(11)]
                                     ].values.sum(axis=1)
    df['coastal_flooded_total'] = df[[f'coastal_{n}_total' for n in ["fm", "gc", "both"]]
                                    ].values.sum(axis=1)

    for f in ["", "flooded_"]:
        df[f'coastal_{f}fm_p'] = df['coastal_fm_total'] / df[f'coastal_{f}total']
        for n in ['both', 'gc']:
            df[f'coastal_{f}{n}_p'] = df[f'coastal_{n}_total'] / df[f'coastal_{f}total']
            for i in range(11):
                df[f'coastal_{f}{n}{i}_p'] = df[f'coastal_{n}{i}_total'] / df[f'coastal_{f}total']

    df['coastal_neither_p'] = 1 - df[
        [f'coastal_{n}_p' for n in ['both', 'fm', 'gc']]
    ].values.sum(axis=1)
    df['fm_total'] = df['coastal_fm_total'] + df['coastal_both_total'] + df['cutoff_fm_total']
    df['fm_cutoff_p'] =  df['cutoff_fm_total'] / df['fm_total']
    df['coastal_flooded_fm+_p'] = df['coastal_flooded_fm_p'] + df['coastal_flooded_both_p']
    df['coastal_flooded_gc+_p'] = df['coastal_flooded_gc_p'] + df['coastal_flooded_both_p']

    df["cell_area"] = (
        (ONE_LAT_KM * df['res_as'] / (60 * 60))**2
        * np.cos(np.radians(0.5 * (df["lat_min"] + df["lat_max"])))
    )
    for suffix in ["", "_flooded", "_fm", "_both", "_gc"]:
        df[f'coastal{suffix}_area'] = df[f'coastal{suffix}_total'] * df["cell_area"]

    rainf_keys = ["ERA5-combined", "WFDE5", "GPCC"]
    for key in rainf_keys:
        df[f"rainf_{key}"] = np.nan
    df['maxwind'] = np.nan
    for idx, row in df.iterrows():
        map_id = row['map_id']
        path = u_const.RAINFALL_MAPS_DIR / source / f"{map_id}.csv"
        rainf_df = pd.read_csv(path)
        rainf_evt_df = rainf_df.iloc[:21]
        rainf_clim_df = rainf_df.iloc[21:]
        n_clim_years = rainf_clim_df.shape[0] / 21
        for key in rainf_keys:
            mean = rainf_clim_df[key].sum() / n_clim_years
            df.loc[idx, f"rainf_{key}"] = rainf_evt_df[key].sum()
            df.loc[idx, f"rainf_{key}_rel"] = rainf_evt_df[key].sum() / mean

        path = u_const.WINDS_DIR / source / f"{map_id}.tif"
        with rasterio.open(path, "r") as src:
            data = src.read(1)
            df.loc[idx, 'maxwind'] = data.max()

    df['dem_comment'] = ""
    for map_id, comment in DEM_COMMENT.items():
        df.loc[df['map_id'] == map_id, 'dem_comment'] = comment

    sl_df = pd.read_csv(u_const.GEOCLAW_DIR / source / "base_sea_level.csv")
    sl_df.columns = [c if c == "map_id" else f"sea_level_{c}" for c in sl_df.columns]
    df = df.merge(sl_df, left_on="map_id", right_on="map_id")
    return df.sort_values(by=['coastal_flooded_fm_p']).reset_index(drop=True)


def main():
    sources = ['dfo', 'gfd', 'rapid']

    parser = argparse.ArgumentParser(description='Compare flood maps with GeoClaw model runs.')
    parser.add_argument(
        'source_m',
        type=str,
        metavar="SOURCE_M",
        choices=(
            sources + [f"{s1}_{s2}" for s1 in sources for s2 in sources if s2 != s1]
        ),
        help='The flood map source (with mask, if specified).',
    )
    parser.add_argument('--thresh', type=float, metavar="THRESHOLD", default=0.0,
                        help='The threshold for GC flooding to be considered 0.')
    parser.add_argument('--zos', type=str, metavar="SEA_LEVEL", default="aviso-fes_max",
                        choices=[f"{sl}-fes_{fes}" for sl in ["0", "aviso", "mercator"]
                                 for fes in ["no", "min", "mean", "max"]],
                        help='The sea level data set used in the GC runs.')
    parser.add_argument('--pluvial', type=str, metavar="PLUVIAL", default="without",
                        choices=PLUVIAL_MODES,
                        help='The pluvial flooding dataset to compliment GC maps.')
    args = parser.parse_args()

    compare_dir = u_const.COMPARE_DIR / args.source_m / args.pluvial / args.zos
    compare_dir.mkdir(parents=True, exist_ok=True)
    generate_compare_tifs(args.source_m, args.zos, args.pluvial, args.thresh)
    df = compute_compare_stats(args.source_m, args.zos, args.pluvial, args.thresh)

    outpath = compare_dir / f"stats-thresh_{args.thresh:.1f}.csv"
    print(f"Writing to {outpath} ...")
    df.to_csv(outpath, index=False)


if __name__ == "__main__":
    main()
