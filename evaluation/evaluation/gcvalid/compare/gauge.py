"""
In the output directory, generate gauge comparison data:
1. Restrict to tide gauges that are at most 5 km in land, and within 0.5 degrees of the compare areas.
2. Restrict to tide gauges where the containing or some of the directly neighboring grid cells have satellite altimetry records.
3. Extract temporal range according to wind exposure (17.5 m/s ±12 hours).
4. Restrict to tide gauges that have continuous valid records for the whole period (of the wind exposure). At most 1 hour missing per day.
5. Vertically reference the gauge and GeoClaw time series.
6. Assign a CoDEC station to each of the observational stations (max. distance 0.1 degrees).
7. Store the preprocessed data.
8. Compare GeoClaw and CoDEC to the other gauge sources, store the fit statistics in a separate file.
"""
import argparse
import pickle

from climada.util.config import CONFIG as CLIMADA_CONFIG
from climada.util.constants import SYSTEM_DIR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

import gcvalid.util.constants as u_const
import gcvalid.util.gauge as u_gauge
import gcvalid.util.io as u_io


DIST_TO_COAST_NASA_TIF = SYSTEM_DIR / CLIMADA_CONFIG.util.coordinates.dist_to_coast_nasa_tif.str()
"""Raster with distance to coast in km (positive off shore, and negative on land)"""

ALTIMETRY_SRC = "aviso"
"""The altimetry source to use"""

AFFECTED_PAD = 0.5
"""Padding (in degrees) around compare area for gauge station to be affected"""

MAX_ALTIMETRY_PAD = 0.375
"""Maximum padding to use in matching with altimetry product (in degrees)"""

MAX_DIST2COAST_INLAND = 5
"""Maximum distance to coast for inland stations (in km)"""

MIN_WIND_SPEED = 17.5
"""Mininum wind speed attained at gauge location (in m/s)"""

MAX_DIST_CODEC = 0.1
"""Maximum distance between codec and reference station (in degrees)"""


def set_affected(gdata):
    gdata_aff = [stdata for stdata in gdata if stdata['affected_bounds']]
    locations_aff = [stdata['location'][::-1] for stdata in gdata_aff]

    with rasterio.Env(VRT_SHARED_SOURCE=False):
        with rasterio.open(DIST_TO_COAST_NASA_TIF, "r") as src:
            for stdata, v in zip(gdata_aff, src.sample(locations_aff)):
                stdata['dist2coast'] = v[0]
                stdata['affected'] = (
                    stdata['wind'] is not None
                    and stdata['wind'].intensity.max() >= MIN_WIND_SPEED
                    and stdata['dist2coast'] >= -MAX_DIST2COAST_INLAND
                )

    for stdata in gdata:
        stdata['affected_w_valid_altimetry'] = (
            stdata['affected'] and stdata['valid_altimetry'])


def restrict_to_wind_exposure(stdata):
    """Restrict records to wind exposed period"""
    if not stdata['affected']:
        stdata['tide_range'] = np.inf
        stdata['referenced'] = None
        return

    stdata['tide_range'] = stdata['tide_levels'].max() - stdata['tide_levels'].min()

    # extract temporal range (from first till last time of wind exposure ±12 hours)
    idx_exposed = (stdata['wind'].intensity.values >= MIN_WIND_SPEED).nonzero()[0]
    t_start, t_end = stdata['wind'].index[idx_exposed[[0, -1]]]
    t_pad = np.timedelta64(12, 'h')
    t_start, t_end = (t_start - t_pad, t_end + t_pad)
    stdata['wind_period'] = (t_start, t_end)

    # restrict 'anomaly' and 'referenced' to wind exposed period
    t_mask = (stdata['anomaly'].index >= t_start) & (stdata['anomaly'].index <= t_end)
    stdata['anomaly'] = stdata['anomaly'][t_mask].dropna()
    stdata['referenced'] = stdata['referenced'][t_mask].dropna()

    n_hour_steps = round(1 + (t_end - t_start) / np.timedelta64(1, 'h'))
    stdata['valid_hour_steps'] = n_hour_steps >= 24
    for i in range(n_hour_steps - 23):
        t0 = t_start + np.timedelta64(i, 'h')
        t1 = t0 + np.timedelta64(24, 'h')
        t_mask = (stdata['referenced'].index >= t0) & (stdata['referenced'].index < t1)
        if t_mask.sum() < 23:
            # at most one hour may be missing per day of data
            stdata['valid_hour_steps'] = False

    stdata['valid'] = (
        stdata['valid_altimetry'] and stdata['valid_hour_steps']
    )


def set_simulated_geoclaw(source, map_id, gdata):
    """Generate a 'simulated' entry for the best GeoClaw record"""
    for stdata in gdata:
        if not stdata['affected'] or stdata['geoclaw'] is None:
            continue

        for gd in stdata['geoclaw']:
            if 'overlap_idx' in gd:
                # It's possible that we have already processed this GeoClaw output because the
                # same GeoClaw output might have been assigned to different reference stations.
                # For example, because two reference stations have exactly or almost the same
                # location (esp. when a single station is included in different sources).
                continue

            # select the record with the largest overlap with the wind exposure period
            wperiod = stdata['wind_period']
            sl_masked = [
                sl[(sl.index >= wperiod[0]) & (sl.index <= wperiod[1])]
                for sl in gd['referenced']
            ]
            overlap_times = [
                sl.index[-1] - sl.index[0] if sl.size > 0 else np.timedelta64(0, 'h')
                for sl in sl_masked
            ]
            gd['overlap_idx'] = np.argmax(overlap_times)
            gd['referenced'] = gd['referenced'][gd['overlap_idx']]
            gd['valid_topo_height'] = gd['topo_height'][gd['overlap_idx']] <= 0

        stdata['simulated'].extend([
            {
                'model': f"geoclaw-zos_{gd['zos']}",
                'valid_topo_height': gd['valid_topo_height'],
                'referenced': gd['referenced'],
                'location': gd['location'],
                'dist': np.linalg.norm(
                    np.array(gd['location']) - np.array(stdata['location'])
                ),
            }
            for gd in stdata['geoclaw']
        ])


def set_simulated_codec(map_id, gdata):
    """Associate a CoDEC station to each"""
    simdata = [
        stdata for stdata in gdata
        if stdata['gsrc'] == "codec" and stdata['affected']
    ]
    if len(simdata) == 0:
        print(f"No CoDEC data for {map_id}!")
        return

    locations = [np.array(sd['location']) for sd in simdata]
    for stdata in gdata:
        if stdata['gsrc'] in ["codec", "gtsm"] or not stdata['affected']:
            continue
        latlon = np.array(stdata['location'])
        dists = [np.linalg.norm(loc - latlon) for loc in locations]
        argmin_dists = np.argmin(dists)
        min_dist = dists[argmin_dists]
        simdata_st = simdata[argmin_dists]
        stdata['simulated'].append({
            'model': "codec",
            'dist': min_dist,
            'referenced': simdata_st['referenced'],
            'location': simdata_st['location'],
        })


def prepare_gaugedata(source, map_id, year, bounds, df_annual_msl):
    out_path = u_const.COMPARE_DIR / source / "gauges" / f"{map_id}.pickle"
    if out_path.exists():
        print(f"Reading {out_path} ...")
        with out_path.open("rb") as fp:
            return pickle.load(fp)

    gdata = u_gauge.load_gaugedata(
        source, map_id, by_gsrc=False, referenced=True,
        geoclaw_zos=f"{ALTIMETRY_SRC}-fes*",
    )

    for stdata in gdata:
        stname = stdata['filename']
        loc = stdata['location'][::-1]
        stdata['gc_loc_misfit'] = np.linalg.norm(
            np.array(stdata['location']) - np.array(stdata['gc_location']))
        stdata['affected_bounds'] = (
            bounds[0] - AFFECTED_PAD <= loc[0] <= bounds[2] + AFFECTED_PAD
            and bounds[1] - AFFECTED_PAD <= loc[1] <= bounds[3] + AFFECTED_PAD)
        stdata['dist2coast'] = np.inf
        stdata['affected'] = False
        stdata['altimetry_pad'] = df_annual_msl[f"{stname}_pad"].values[0]
        stdata['valid_altimetry'] = stdata['altimetry_pad'] <= MAX_ALTIMETRY_PAD
        stdata['valid_hour_steps'] = False
        stdata['valid'] = False
        stdata['simulated'] = []
        stdata['tide_levels_rmse'] = np.sqrt(np.mean(
            (stdata['tide_levels_full'].values - stdata['tide_levels'].values)**2
        ))

    set_affected(gdata)
    if any(stdata['affected'] for stdata in gdata):
        for stdata in gdata:
            restrict_to_wind_exposure(stdata)
        set_simulated_geoclaw(source, map_id, gdata)
        set_simulated_codec(map_id, gdata)

    # reduce data by restricting to data that is relevant for comparison
    keys = [
        'gsrc', 'map_id', 'filename', 'location', 'gc_location', 'gc_loc_misfit', 'dist2coast',
        f'annual_msl_{ALTIMETRY_SRC}', 'valid_altimetry', 'valid_hour_steps', 'valid', 'wind',
        'affected_bounds', 'affected', 'affected_w_valid_altimetry', 'referenced', 'simulated',
        'tide_levels_rmse',
    ]
    gdata = [
        {key: stdata[key] for key in keys}
        for stdata in gdata
    ]

    print(f"Writing to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        pickle.dump(gdata, fp)
    return gdata


def prepare_meta(source):
    compare_ds = pd.concat([
        u_io.read_compare_df(
            source, "without", f"{ALTIMETRY_SRC}-fes_{fes}", 0.0,
            apply_filters=False, verbose=False,
        ).sort_values(by=["map_id"]).reset_index(drop=True)
        for fes in ["min", "mean", "max", "no"]
    ]).set_index(["zos", "map_id"]).to_xarray()
    compare_df = compare_ds.isel(zos=0).to_dataframe().reset_index()
    for coord in ["lon", "lat"]:
        for op in ["min", "max"]:
            col = f"{coord}_{op}"
            compare_df[col].values[:] = getattr(compare_ds[col], op)(
                dim=["zos"], skipna=True,
            ).values
    compare_df['width'] = compare_df['lon_max'] - compare_df['lon_min']
    compare_df['height'] = compare_df['lat_max'] - compare_df['lat_min']
    compare_df['area'] = compare_df['width'] * compare_df['height']
    compare_df['coastal_area'].values[:] = compare_ds['coastal_area'].max(dim=["zos"]).values

    fm_meta_path = u_const.FLOODMAPS_DIR / source / "meta.hdf5"
    fm_meta = pd.read_hdf(fm_meta_path).sort_values(by="map_id")
    fm_meta = fm_meta[np.isin(fm_meta['map_id'], compare_df['map_id'])].reset_index(drop=True)

    return fm_meta, compare_df


def compute_fit_quality(gaugedata):
    gaugedata = [
        stdata for stdata in gaugedata
        if stdata['affected'] and stdata['valid']
    ]
    fit_data = []
    for stdata in gaugedata:
        st_series = stdata['referenced']
        st_series.name = "measured"
        for simdata in stdata['simulated']:
            model = simdata['model']
            sim_series = simdata['referenced']
            sim_series.name = "simulated"

            joined = pd.DataFrame(sim_series).join(st_series, how="inner")
            sim_max, st_max = joined.max()
            sim_mean, st_mean = joined.mean()
            st_range = st_max - joined['measured'].min()
            sim_range = sim_max - joined['simulated'].min()

            joined -= (sim_mean, st_mean)
            rmse = np.nan
            pearson = np.nan
            if joined.shape[0] > 0:
                rmse = ((joined.values[:, 0] - joined.values[:, 1])**2).mean()**0.5
                pearson = joined.corr(method='pearson').values[0, 1]

            sim_valid = (
                simdata['valid_topo_height']
                if model.startswith("geoclaw")
                else simdata['dist'] <= MAX_DIST_CODEC,
            )

            fit_data.append(pd.DataFrame({
                "stname": stdata['filename'],
                "map_id": stdata['map_id'],
                "gsrc": stdata['gsrc'],
                "lat": stdata['location'][0],
                "lon": stdata['location'][1],
                "model": model,
                "valid": sim_valid,
                "sim_dist_deg": simdata['dist'],
                "max_obs": st_max,
                "max_sim": sim_max,
                "range_obs": st_range,
                "range_sim": sim_range,
                "mean_obs": st_mean,
                "mean_sim": sim_mean,
                "tl_rmse": stdata['tide_levels_rmse'],
                "rmse": rmse,
                "pearson": pearson,
            }, index=[0]))
    fit_df = pd.concat(fit_data)

    fit_df['model_gc'] = fit_df['model'].str.startswith("geoclaw")
    fit_df['dmax_signed'] = fit_df["max_obs"] - fit_df["max_sim"]
    fit_df['dmax'] = np.abs(fit_df['dmax_signed'])
    fit_df['dmaxrel_signed'] = fit_df['dmax_signed'] / fit_df["max_obs"]
    fit_df['dmaxrel'] = np.abs(fit_df['dmaxrel_signed'])
    fit_df['dmean_signed'] = fit_df["mean_obs"] - fit_df["mean_sim"]
    fit_df['dmean'] = np.abs(fit_df['dmean_signed'])
    fit_df['drange'] = np.abs(fit_df["range_obs"] - fit_df["range_sim"])
    fit_df['drange_rel'] = fit_df['drange'] / np.fmax(fit_df["range_obs"], 1e-5)

    fit_df['fit_quality'] = (
        fit_df['dmean'] / 1000 + np.fmin(1, fit_df['drange'] / 1000) * fit_df['drange_rel']
    )

    return fit_df


def main():
    parser = argparse.ArgumentParser(description='Compare GeoClaw and CoDEC to gauge sources.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    fm_meta, compare_df = prepare_meta(source)

    df_annual_msl = pd.read_hdf(u_const.GAUGES_DIR / f"annual_msl_{ALTIMETRY_SRC}.hdf5")
    df_annual_msl = df_annual_msl.set_index('years')

    gaugedata = []
    for idx, row in fm_meta.iterrows():
        map_id = row['map_id']
        compare_row = compare_df[compare_df['map_id'] == map_id].iloc[0]
        compare_bounds = (
            compare_row['lon_min'], compare_row['lat_min'],
            compare_row['lon_max'], compare_row['lat_max'],
        )
        gaugedata.extend(
            prepare_gaugedata(source, map_id, int(row['date'][:4]), compare_bounds, df_annual_msl)
        )

    print()
    gsources = sorted(set(stdata['gsrc'] for stdata in gaugedata))
    for gsrc in gsources:
        gdata_sub = [stdata for stdata in gaugedata if stdata['gsrc'] == gsrc]
        print(gsrc)
        print(f"Total number of gauges: "
              f"{len(set(stdata['filename'] for stdata in gdata_sub))} "
              f"with {len(gdata_sub)} records, "
              f"{len(set(gd['map_id'] for gd in gdata_sub))} flood maps")
        for predicate in ['affected', 'affected_w_valid_altimetry', 'valid']:
            print(f"Number of {predicate} gauges: "
                  f"{len(set(gd['filename'] for gd in gdata_sub if gd[predicate]))} "
                  f"with {sum([gd[predicate] for gd in gdata_sub])} records, "
                  f"{len(set(gd['map_id'] for gd in gdata_sub if gd[predicate]))} flood maps")
        print()

    fit_df = compute_fit_quality(gaugedata)
    path = u_const.COMPARE_DIR / source / "gauges" / "stats.csv"
    print(f"Writing to {path} ...")
    fit_df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
