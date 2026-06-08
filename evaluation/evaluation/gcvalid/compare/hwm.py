
import argparse
import json

import climada.util.coordinates as u_coord
import numpy as np
import pandas as pd
import rasterio

import gcvalid.util.constants as u_const
import gcvalid.util.io as u_io


def _hwms_from_compare_map(compare_tif_path, all_hwms, fev_events):
    map_id = compare_tif_path.name.split("-thresh")[0]
    ibtracs_id = map_id.split("-")[0]
    if ibtracs_id not in fev_events:
        return []
    fev_name = fev_events[ibtracs_id]

    with rasterio.open(compare_tif_path, "r") as src:
        bounds = src.bounds

    qualities = [
        # Optionally, exclude some of these
        'Excellent: +/- 0.05 ft',
        'Good: +/- 0.10 ft',
        'Fair: +/- 0.20 ft',
        'Poor: +/- 0.40 ft',
        'VP: > 0.40 ft',
        'Unknown/Historical',
    ]

    sel = []
    for hwm in all_hwms:
        if hwm['eventName'] != fev_name:
            continue
        hwm['ibtracs_id'] = ibtracs_id
        hwm['map_id'] = map_id
        has_height_info = (
            hwm['height_above_gnd_m'] != "" and hwm['height_above_gnd_m'] > 0
            or hwm['elev_m'] != "" and hwm['elev_m'] > 0
        )
        if not has_height_info:
            continue
        if hwm['height_above_gnd_m'] == "":
            hwm['height_above_gnd_m'] = np.nan
        if hwm['hwmQualityName'] not in qualities:
            continue
        if (
            hwm['longitude'] < bounds.left or hwm['longitude'] > bounds.right
            or hwm['latitude'] < bounds.bottom or hwm['latitude'] > bounds.top
        ):
            continue
        sel.append(hwm)

    hwm_locations = [
        (hwm['longitude'], hwm['latitude']) for hwm in sel
    ]
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        with rasterio.open(compare_tif_path, "r") as src:
            # "sample" uses nearest neighbor interpolation
            is_coastal = [v[0] < 50 for v in src.sample(hwm_locations)]
    sel = [hwm for i, hwm in enumerate(sel) if is_coastal[i]]

    return sel


def _sample_from_dem(source, map_id, lats, lons):
    # sample from the 30as-elevation data with nearest neighbor interpolation
    locations = list(zip(lons, lats))
    path = u_const.ELEVATION_MAPS_DIR / source / f"{map_id}.tif"
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        with rasterio.open(path, "r") as src:
            return [v[0] for v in src.sample(locations)]


def _hwm_add_flooded_status(sel, source):
    # sample flooded status from flood maps
    if len(sel) == 0:
        return

    map_id = sel[0]['map_id']
    ibtracs_id = sel[0]['ibtracs_id']
    paths = {
        'floodmap': u_const.FLOODMAPS_DIR / source / "clean_by_sid" / f"{map_id}.tif",
        'bt_climada': u_const.BATHTUB_DIR / "climada" / source / f"{map_id}.tif",
        'bt_aq_codec': (
            u_const.BATHTUB_DIR / "codec" / source / "aqueduct_output" / map_id / "inun.tif"
        ),
        'max_wind': u_const.WINDS_DIR / source / f"{map_id}.tif",
        'runoff': u_const.RUNOFF_MAPS_DIR / source / f"{map_id}.tif",
    }
    for prot in ["noprot", "2yprot", "flopros"]:
        paths[f"cama_{prot}"] = (
            u_const.PLUVIAL_MAPS_DIR / f"isimip3a{prot}" / source / f"{map_id}.tif"
        )
    for fes in ["min", "mean", "max", "no"]:
        paths[f"geoclaw-fes_{fes}"] = (
            u_const.GEOCLAW_DIR / source / "results"
            / f"{ibtracs_id}_{source}-zos_aviso-fes_{fes}.tif"
        )
        bt_dir = u_const.BATHTUB_DIR / f"geoclaw-fes_{fes}" / source
        paths[f"bt_aq_geoclaw-fes_{fes}"] = bt_dir / "aqueduct_output" / map_id / "inun.tif"
    res = 5 / 3600
    pad = 15 / 3600
    with rasterio.Env(VRT_SHARED_SOURCE=False):
        for key, path in paths.items():
            # After the filtering in _hwms_from_compare_map, all HWM locations are actually within
            # the areas where flood maps are available. However, due to the padding that we apply,
            # the following step might mix in locations with missing values or permanent water.
            for hwm in sel:
                padded_bounds = (
                    hwm['longitude'] - pad, hwm['latitude'] - pad,
                    hwm['longitude'] + pad, hwm['latitude'] + pad,
                )

                if key.startswith("bt_aq") and not path.exists():
                    data = np.full((2, 2), np.nan)
                else:
                    [data], _ = u_coord.read_raster_bounds(
                        path, padded_bounds, res=res, resampling="bilinear",
                    )
                    data = data.astype(float)

                if key.startswith("bt_aq"):
                    data[np.isnan(data)] = 0.0
                elif key == "floodmap":
                    data[data == 255] = np.nan
                elif key not in ["max_wind", "runoff"]:
                    data[(data > 10) | (data < 0)] = np.nan
                if np.isnan(data).all():
                    print("All NaNs:", map_id, key, hwm["longitude"], hwm["latitude"])
                    hwm[key] = np.nan
                else:
                    hwm[key] = np.nanmean(data)

        lats = np.array([hwm['latitude'] for hwm in sel])
        lons = np.array([hwm['longitude'] for hwm in sel])
        elevs = _sample_from_dem(source, map_id, lats, lons)
        dists = u_coord.dist_to_coast_nasa(lats, lons, highres=True, signed=True)
        for e, d, hwm in zip(elevs, dists, sel):
            hwm["dist2coast"] = float(d)
            hwm["dem"] = float(e)


def read_hwms(source, maps=None, add_flooded_status=False, as_dataframe=False):
    """Select all HWMs that are within compare areas

    Parameters
    ----------
    source : str
        Floodmap source.
    maps : array-like, optional
        If given, only read the HWM data for the specified maps. Default: None
    add_flooded_status : bool, optional
        If True, include flood information from additional sources, such as CaMa or GeoClaw.
        Default: False
    as_dataframe : bool, optional
        If True, return data as DataFrame. Default: False

    Returns
    -------
    list of dicts, or pd.DataFrame (depending on `as_dataframe`)
    """
    with u_const.HWMS_FILE.open("r") as fp:
        high_water_marks = json.load(fp)
    fev_events = {d['ibtracs_id']: d['eventName'] for d in high_water_marks}

    def_zos = "aviso-fes_max"
    def_pluvial = "without"
    if maps is None:
        compare_df = u_io.read_compare_df(source, def_pluvial, def_zos, 0)
        maps = compare_df.loc[np.isin(compare_df["ibtracs_id"], list(fev_events.keys())), "map_id"]

    selection = []
    for map_id in maps:
        fstem = f"{map_id}-thresh_0.0"
        compare_dir = u_const.COMPARE_DIR / source / def_pluvial / def_zos
        outpath = u_const.COMPARE_DIR / source / "hwms" / f"{fstem}.json"
        if not outpath.exists():
            sel = _hwms_from_compare_map(
                compare_dir / f"{fstem}.tif", high_water_marks, fev_events)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with outpath.open("w") as fp:
                json.dump(sel, fp)
        else:
            with outpath.open("r") as fp:
                sel = json.load(fp)

        if len(sel) == 0:
            continue

        if add_flooded_status and 'floodmap' not in sel[0]:
            _hwm_add_flooded_status(sel, source)
            with outpath.open("w") as fp:
                json.dump(sel, fp)

        selection.extend(sel)

    # 2023-12-15: Tried to extract a "coastal" classification feature from the "waterbody" field
    # by looking for occurrence of "atlantic" or "ocean". This removed 96% of the HWMs, and left
    # only two storm events (35 HWMs). Maximum height above ground among these was 0.94m and for
    # 18 among them, DEM error was larger than inundation height. GeoClaw and floodmaps both had
    # a hit rate of over 80% (Aqueduct slightly less, CLIMADA ~60%).
    # Found that there actually is a Coastal/Riverine classification feature "hwm_environment" that
    # is really given for every single HWM in the data base.

    if not as_dataframe:
        return selection

    columns = [
        "hwm_id", "map_id", "ibtracs_id", "longitude", "latitude", "height_above_gnd_m", "elev_m",
        "hwmQualityName", "hwm_environment",
    ]
    if add_flooded_status:
        columns.extend([
            "dist2coast", "dem", "max_wind", "runoff", "bt_climada", "bt_aq_codec", "floodmap",
        ] + [
            f"cama_{prot}" for prot in ["noprot", "2yprot", "flopros"]
        ] + [
            f"{bt_prefix}geoclaw-fes_{fes}"
            for fes in ["min", "mean", "max", "no"]
            for bt_prefix in ["", "bt_aq_"]
        ])

    df = pd.DataFrame({c: np.array([hwm[c] for hwm in selection]) for c in columns})
    df["fm_source"] = source
    return df
