"""
For each floodmap event, list all (gtsm/gesla/...) gauges within the floodmap area
that have records at the given date.
"""
import argparse
import io
import pickle
import re
import time
import warnings
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import scipy.io
import shapely.geometry
import xarray as xr

import gcvalid.util.constants as u_const
import gcvalid.util.gauge as u_gauge


UHSLC_DIR = u_const.GAUGES_DIR / "uhslc_rqds"

WSL_DIR = u_const.GAUGES_DIR / "wsl_data"

GESLA3_DIR = u_const.GAUGES_DIR / "gesla3_data"

WSL_INDEX_URL = "https://webcritech.jrc.ec.europa.eu/worldsealevelinterface/Default.aspx?list=true"

WSL_INDEX_CSV = u_const.GAUGES_DIR / "wsl_gauges.csv"

ALL_KEYS = [
    "discarded",
    "waterlevel",
    "harmonics",
    "harmonics_90d",
    "annual_msl",
    "annual_msl_detided",
    "annual_msl_detided_90d",
]
"""Keys that are set (if all are set already, nothing is done)"""

GTSM_ALL_KEYS = [
    "discarded",
    "waterlevel",
    "combined",
    "annual_msl",
    "annual_msl_detided",
]
"""For GTSM/CoDEC, different keys are set"""


def codec_meta_df():
    df = (
        xr.open_dataset(u_const.CODEC_COORD_FILE)
        .to_dataframe()
        .reset_index(drop=True)
        .rename(columns={
            "station_x_coordinate": "lon",
            "station_y_coordinate": "lat",
        })
        .drop(columns=[
            "__xarray_dataarray_variable__",
            "station_name"
        ])
    )
    df['filename'] = df.index.map(lambda i: f"gtsm_station{i:05d}.nc")
    df['coastal'] = True
    return df


def gtsm_from_date_and_box(data, df, b_box, date, start, end, raw=False):
    src = "codec" if raw else "gtsm"
    df = df[df.within(b_box) & df.coastal]
    date_y_start = np.datetime64(pd.Timestamp(date).strftime("%Y-01-01"))
    date_y_end = np.datetime64(pd.Timestamp(date).strftime("%Y-12-31"))
    changed = False
    for tg_i, (_, row) in enumerate(df.iterrows()):
        print(f"Processing {src}: {tg_i:3d}/{df.shape[0]} ...", end="\r")
        stname = row.filename
        stdata = [d for d in data if d['filename'] == stname]
        if len(stdata) > 1:
            raise Exception(f"{src}: multiple stations with name {stname}")
        elif len(stdata) == 1:
            stdata = stdata[0]
            if all(key in stdata for key in GTSM_ALL_KEYS):
                continue
            if "discarded" in stdata and stdata["discarded"] != False:
                continue
        else:
            stdata = {"filename": row.filename, "location": (row.lat, row.lon), "discarded": False}
            data.append(stdata)
        changed = True

        if "discarded" not in stdata:
            stdata["discarded"] = False

        if raw:
            t_period = (start - np.timedelta64(366, "D"), end + np.timedelta64(366, "D"))

            ds_tides = xr.open_dataset(u_const.CODEC_TIDES_DIR / stname)
            t_mask = ((ds_tides.index >= t_period[0]) & (ds_tides.index <= t_period[1]))
            ds_tides = ds_tides.sel(index=t_mask)

            ds_combined = xr.open_dataset(u_const.CODEC_COMBINED_DIR / stname)
            t_mask = ((ds_combined.index >= t_period[0]) & (ds_combined.index <= t_period[1]))
            ds_combined = ds_combined.sel(index=t_mask)

            ds_tides['combined'] = ds_combined.waterlevel
            tg = ds_tides.to_dataframe()[["waterlevel", "combined"]]
        else:
            tg = pd.read_hdf(u_const.GSLCOMP_GTSM_DIR / f"{stname}.h5")
            tg = tg.rename(columns={"gtsm_tide_slr": "waterlevel"})

        if ((tg.index[:1] > date) | (date > tg.index[-1:]))[0]:
            stdata["discarded"] = "no data in event period"
            continue

        if "annual_msl" not in stdata:
            tg_y = tg['combined'][date_y_start:date_y_end]
            tg_y_detided = tg_y - tg['waterlevel'][date_y_start:date_y_end]
            stdata["annual_msl"] = tg_y.mean()
            stdata["annual_msl_detided"] = tg_y_detided.mean()

        if "waterlevel" not in stdata:
            stdata["waterlevel"] = tg['waterlevel'][start:end]

        if "combined" not in stdata:
            stdata["combined"] = tg['combined'][start:end]
    print(f"Processing {src}: {df.shape[0]}/{df.shape[0]} ...")
    return changed


def gesla_read_data_file(name):
    path = u_const.GSLCOMP_GESLA_DIR / f"{name}.mat"
    mat = scipy.io.loadmat(path)
    if mat['AMSL'].shape[1] == 0:
        return None
    # 719529 : number of days from 0000-00-00 till 1970-01-01 in the proleptic ISO calendar
    t = pd.to_datetime(mat["Thour"][:, 0] - 719529, unit="D").round("H")
    whour_detr = pd.Series(mat["Whour_detr"][:, 0], index=t)
    amsl = pd.Series(
        mat["AMSL"][:, 1],
        index=pd.to_datetime(mat["AMSL"][:, 0].astype(int).astype(str)),
    )
    tg_df = pd.DataFrame({"whour_detr": whour_detr, "amsl": amsl}, index=t)
    return tg_df["whour_detr"] + tg_df["amsl"].interpolate(method="ffill").fillna(0)


def wsl_meta_df():
    if not WSL_INDEX_CSV.exists():
        [df] = pd.read_html(WSL_INDEX_URL)
        splitted = df["lat/lon"].str.split("/", n=1, expand=True)
        df["lat"], df["lon"] = splitted[0], splitted[1]
        df.to_csv(WSL_INDEX_CSV, index=None)
    return pd.read_csv(WSL_INDEX_CSV)


def wsl_download_records(station_id, d_start, d_end, retry=True, new_api=True):
    warn_prefix = f"Warning ({station_id}, {d_start}, {d_end})"
    if 100219 <= int(station_id) <= 100239:
        # enforce the old API for INCOIS buoys
        new_api = False

    dt_start = pd.to_datetime(str(d_start))
    dt_end = pd.to_datetime(str(d_end))
    str_start = dt_start.strftime('%d %b %Y 00:00')
    str_end = dt_end.strftime('%d %b %Y 23:59')
    # a ridiculously high number to make sure we get the original time resolution
    nrec = int(1e9)

    if new_api:
        base_url = f"https://webcritech.jrc.ec.europa.eu/SeaLevelsDb/api/Data/Get/{station_id}"
        payload = dict(tMin=str_start, tMax=str_end, nRec=nrec, mode="txt")
    else:
        base_url = "https://webcritech.jrc.ec.europa.eu/worldsealevelinterface/Default.aspx"
        payload = dict(id=station_id, tmin=str_start, tmax=str_end, nrec=nrec)

    r = requests.get(base_url, params=payload)

    if r.status_code == 404:
        print(f"{warn_prefix}: not found (404)")
        df = pd.DataFrame({"time": [], "wat_level": []})
        df['time'] = pd.to_datetime(df['time'], dayfirst=True)
        df = df.set_index("time")
        harmonics = None
        return harmonics, df
    csv_str = r.text
    if "<!DOCTYPE html>" in csv_str:
        print(f"{warn_prefix}: server error string in result")
        csv_str = csv_str.split("<!DOCTYPE html>")[0]
    if csv_str == "Buoy is not enabled for current user":
        print(f"{warn_prefix}: {csv_str}")
        return None, None
    csv_io = io.StringIO(csv_str)

    if new_api:
        harmonics_start = "#  HARMONICS="
        harmonics_end = "\n"
    else:
        harmonics_start = "# Harmonics=HARMONICS_WE:"
        harmonics_end = "|\n"
    harmonics = None
    for line in csv_io:
        if line.startswith(harmonics_start):
            line = line.replace("\r\n", "\n")
            harmonics = line.replace(harmonics_start, "").replace(harmonics_end, "")
            break

    if harmonics is None:
        # If no harmonics have been found in the output, the server probably sent an
        # error output. In that case, we retry once with the new, and twice with the old API.
        if not retry and not new_api:
            # If erroneous output prevails after 4 attempts, raise an exception.
            print(base_url)
            print(payload)
            print(csv_str)
            raise Exception(
                "Failed to download after retry with new and old api (see output above)!")
        time.sleep(1.0)
        return wsl_download_records(
            station_id, d_start, d_end,
            retry=new_api and not retry,
            new_api=new_api and retry)
    harmonics = np.array([
        [float(val) for val in h.split(",")]
        for h in harmonics.split("|")
    ])

    columns = ["time", "wat_level", "tide", "level-tide"] if new_api else ["time", "wat_level"]
    df = pd.read_csv(csv_io, comment="#", names=columns)
    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    df = df.set_index("time")
    return harmonics, df


def wsl_remove_NaT(df):
    df['defective'] = df.index.isna()
    if df['defective'].sum() > 0:
        defective_idx = df['defective'].values.nonzero()
        for idx in defective_idx:
            defective_months = []
            if idx > 0:
                defective_months.append(df.index[idx - 1])
            if df.index.size > idx + 1:
                defective_months.append(df.index[idx + 1])
            for date in defective_months:
                df['defective'] = (
                    (df.index.year == date.year) &
                    (df.index.month == date.month))
        df = df[~df['defective']]
    return df.drop(columns=['defective'])


def wsl_download_month_by_month(station_id, d_start, d_end, offline=False):
    path = WSL_DIR / f"{station_id}.csv"
    path_harmonics = WSL_DIR / f"{station_id}.npz"
    columns = ["wat_level"]
    df = pd.DataFrame([], columns=columns, index=[])
    harmonics = None
    if path.exists():
        df = wsl_remove_NaT(pd.read_csv(path, index_col=0, parse_dates=True).dropna()).sort_index()
        with np.load(path_harmonics, allow_pickle=False) as npzfile:
            harmonics = npzfile['harmonics']
    affected_days = pd.DatetimeIndex(pd.date_range(d_start, d_end, freq="D"))
    affected_months = np.unique(np.stack((affected_days.month, affected_days.year), axis=-1), axis=0)
    downloaded = []
    for month, year in affected_months:
        month_start = pd.Timestamp(f"{year:04d}-{month:02d}-01")
        month_end = month_start + pd.tseries.offsets.MonthEnd(0)
        if df[month_start:month_end].size > 0:
            continue
        if offline:
            print("wsl: offline mode, won't download", station_id, year, month)
            return None, None
        try:
            new_harmonics, df_month = wsl_download_records(
                station_id, month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))
        except:
            print("wsl: failed to download", station_id, year, month)
            new_harmonics, df_month = None, None
        if df_month is None:
            return None, None
        if new_harmonics is not None:
            harmonics = new_harmonics
        downloaded.append(df_month[columns])
    if len(downloaded) > 0:
        df = pd.concat([df] + downloaded).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        if df.shape[0] > 0 and harmonics is not None:
            df.to_csv(path)
            np.savez(path_harmonics, harmonics=harmonics)
    return harmonics, df[d_start:d_end]


def uhslc_read_data_file(station_id):
    paths = list(UHSLC_DIR.glob(f"*_hourly/h{station_id.lower()}.zip"))
    if len(paths) != 1:
        print(f"Zip-files for station {station_id}: {paths}")
        return None
    path = paths[0]
    tgdata = []
    with zipfile.ZipFile(path) as zfp:
        for fname in zfp.namelist():
            baseyear = {"g": 1800, "h": 1900, "i": 2000}[fname[0]]
            fyear = baseyear + float(fname.split(".")[0][-2:])
            if fyear < 1997:
                continue
            with zfp.open(fname) as fp:
                lines = [l.decode().strip() for l in fp.readlines()]
            header_line, data_lines = lines[0], lines[1:]

            times = [(l[11:15], l[15:17], l[17:19], int(l[19:20])) for l in data_lines]
            times = np.concatenate([
                [np.datetime64(f"{y}-{m}-{d} {h:02d}:00") for h in range(12 * (H - 1), 12 * H)]
                for y, m, d, H in times
            ])

            gmt_offset = float(header_line[71:76].replace(" ", "0")) / 10
            if gmt_offset != 0:
                print(f"Non-zero GMT-offset for station {station_id}: {gmt_offset:.1f}")
                times -= gmt_offset * np.timedelta64(60, 'm')

            values = [l[20:] for l in data_lines]
            values = np.concatenate([
                [float(l[i:i + 5]) for i in range(0, len(l), 5)]
                for l in values
            ])
            values[values == 9999] = np.nan

            if len(times) != len(values):
                print(f"Non-matching lengths of times and values for station {station_id}")
            tgdata.append(pd.Series(values, index=times))
    return pd.concat(tgdata).dropna()


def gesla3_read_data_file(filename):
    with zipfile.ZipFile(GESLA3_DIR / "GESLA3.0_ALL.zip") as zfp:
        with zfp.open(filename) as fp:
            df = pd.read_csv(
                fp,
                skiprows=41,
                names=["date", "time", "sea_level", "qc_flag", "use_flag"],
                sep="\s+",
                parse_dates=[[0, 1]],
                index_col=0,
            )
    duplicates = df.index.duplicated()
    if duplicates.sum() > 0:
        df = df.loc[~duplicates]
        print(f"gesla3: duplicate timestamps in file {filename} removed.")
    return df.sea_level


def gesla3_meta_df():
    df = pd.read_csv(
        GESLA3_DIR / "GESLA3_ALL.csv",
        parse_dates=["START DATE/TIME", "END DATE/TIME"],
    )
    # "GAUGE TYPE" is one of ["Coastal", "Lake", "River"]
    # We restrict our (surge) analysis to coastal gauges!
    df = df[df["GAUGE TYPE"] == "Coastal"].copy()
    return df.rename(columns={"LONGITUDE": "lon", "LATITUDE": "lat"})


def gaugedata_discard_duplicates(data):
    """Mark duplicate tide gauge stations as discarded

    At each location, keep the entry with the highest number of valid data points so that
    there is not more than one tide gauge station within a radius of 1 arc-minute.
    """
    changed = False

    points = pd.DataFrame(
        [
            (
                stdata['filename'],
                stdata['discarded'] != False,
                0 if "waterlevel" not in stdata else stdata['waterlevel'].size,
                stdata['location'][0],
                stdata['location'][1],
            )
            for stdata in data
        ],
        columns=["name", "discarded", "n_valid", "lat", "lon"])
    points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points['lon'], points['lat']))
    points_buff = points.geometry.buffer(60 / 3600)
    selected = set()
    duplicate_of = ["" for stdata in data]
    for idx, row in points.iterrows():
        if row.discarded:
            continue
        overlap_idx = points_buff.contains(row.geometry).values.nonzero()[0]
        sel_idx = overlap_idx[np.argmax(points['n_valid'].values[overlap_idx])]
        sel_name = points.name.values[sel_idx]
        selected.add(sel_idx)
        for i in overlap_idx[overlap_idx != sel_idx]:
            duplicate_of[i] = f"duplicate of {sel_name}"

    for idx, stdata in enumerate(data):
        if duplicate_of[idx] == "" or idx in selected or stdata['discarded'] != False:
            continue
        stdata['discarded'] = duplicate_of[idx]
        changed = True

    for stdata in data:
        if stdata['discarded'] == False:
            continue
        for key in list(stdata.keys()):
            if key not in ["filename", "location", "discarded"]:
                del stdata[key]
                changed = True

    return changed


def gaugedata_from_date_and_box(tg_src, data, df, b_box, date, start, end, offline=False):
    if tg_src in ["gtsm", "codec"]:
        return gtsm_from_date_and_box(data, df, b_box, date, start, end, raw=(tg_src == "codec"))
    elif tg_src == "gesla":
        date_mask = ~df.lon.isna()
    elif tg_src == "wsl":
        date_mask = ~df.lon.isna()
    elif tg_src == "uhslc":
        date_mask = ((df.y_start <= pd.to_datetime(start).year)
                     & (pd.to_datetime(end).year <= df.y_end))
    elif tg_src == "gesla3":
        date_mask = ((df["START DATE/TIME"].dt.year <= pd.to_datetime(start).year)
                     & (pd.to_datetime(end).year <= df["END DATE/TIME"].dt.year))
    df = df[df.within(b_box) & date_mask]
    date_y_start = np.datetime64(pd.Timestamp(date).strftime("%Y-01-01"))
    date_y_end = np.datetime64(pd.Timestamp(date).strftime("%Y-12-31"))
    year_td64 = np.timedelta64(366, "D")
    day_td64 = np.timedelta64(1, "D")
    changed = False
    stname_col = {
        "gesla": "filename",
        "uhslc": "station_id",
        "gesla3": "FILE NAME",
        "wsl": "Description",
    }[tg_src]
    read_data_file_fun = {
        "gesla": gesla_read_data_file,
        "uhslc": uhslc_read_data_file,
        "gesla3": gesla3_read_data_file,
        "wsl": None,
    }[tg_src]
    for tg_i, (_, row) in enumerate(df.iterrows()):
        print(f"Processing {tg_src}: {tg_i:3d}/{df.shape[0]} ...", end="\r")
        stname = row[stname_col]
        stdata = [d for d in data if d['filename'] == stname]
        if len(stdata) > 1:
            raise Exception(f"{tg_src}: multiple stations with name {stname}")
        elif len(stdata) == 1:
            stdata = stdata[0]
            if all(key in stdata for key in ALL_KEYS):
                continue
            if "discarded" in stdata and stdata["discarded"] != False:
                continue
        else:
            stdata = {"filename": stname, "location": (row.lat, row.lon), "discarded": False}
            data.append(stdata)
        changed = True

        if "discarded" not in stdata:
            stdata["discarded"] = False

        if tg_src == "wsl":
            # obtain precomputed harmonics
            wsl_station_id = row['ID'].strip("#")
            harmonics, tg = wsl_download_month_by_month(
                wsl_station_id, start, end, offline=offline)
        else:
            # compute harmonics only when needed
            harmonics = None
            tg = read_data_file_fun(stname)

        if tg is None or tg.size == 0:
            # WARNING: In case of wsl, this can also happen due to a failed download attempt or in offline mode
            # that means that data in "wsl" that is discarded as "no data" might actually have data, but it
            # wasn't available at the time of running this script.
            stdata['discarded'] = "no data"
            continue

        if tg_src == "wsl":
            tg = tg["wat_level"].dropna().resample("1h").mean().dropna()
        else:
            if (start - tg.index[0]) / year_td64 < 1.5:
                tg = tg[:tg.index[0] + 3 * year_td64]
            elif (tg.index[-1] - end) / year_td64 < 1.5:
                tg = tg[tg.index[-1] - 3 * year_td64:]
            else:
                tg = tg[start - 1.5 * year_td64:end + 1.5 * year_td64]
            if tg_src == "gesla3":
                tg[tg == row['NULL VALUE']] = np.nan
            tg = tg.dropna()

        tg_windowed = tg[start:end]
        if tg_windowed.size == 0:
            stdata['discarded'] = "no data in event period"
            continue

        # load 90d series for short-term harmonics
        date_90d_start = start - 45 * day_td64
        date_90d_end = end + 45 * day_td64
        if tg_src == "wsl":
            _, tg_90d = wsl_download_month_by_month(
                wsl_station_id, date_90d_start, date_90d_end, offline=offline)
            if tg_90d is None or tg_90d.shape[0] == 0:
                # not marked as discarded -> retry next time
                print(f"wsl: error while downloading 90d data for {stname} ({wsl_station_id})")
                continue
            tg_90d = tg_90d["wat_level"].dropna().resample("1h").mean().dropna()
        else:
            tg_90d = tg[date_90d_start:date_90d_end]

        if tg_90d.size < 80 * 24:
            # require the record to cover at least 80 full days (for harmonic analysis)
            stdata['discarded'] = "not enough data for harmonic analysis"
            continue

        # load annual series for annual mean
        if tg_src == "wsl":
            _, tg_y = wsl_download_month_by_month(
                wsl_station_id, date_y_start, date_y_end, offline=offline)
            if tg_y is None or tg_y.shape[0] == 0:
                # not marked as discarded -> retry next time
                print(f"wsl: error while downloading annual data for {stname} ({wsl_station_id})")
                continue
            tg_y = tg_y["wat_level"].dropna().resample("1h").mean().dropna()
        else:
            tg_y = tg[date_y_start:date_y_end]

        if tg_y.size < 0.9 * 366 * 24:
            # require the whole record to cover at least 0.9 years (for annual mean)
            stdata['discarded'] = "not enough data for annual mean"
            continue

        if "waterlevel" not in stdata:
            stdata["waterlevel"] = tg_windowed

        if "harmonics" not in stdata:
            if tg_src != "wsl":
                harmonics = u_gauge.fit_harmonics(tg)
            stdata["harmonics"] = harmonics
        harmonics = stdata["harmonics"]

        if "harmonics_90d" not in stdata:
            stdata["harmonics_90d"] = u_gauge.fit_harmonics(tg_90d)
        harmonics_90d = stdata["harmonics_90d"]

        if "annual_msl" not in stdata:
            stdata["annual_msl"] = tg_y.mean()

        if "annual_msl_detided" not in stdata:
            tide_fun = u_gauge.harmonics_fun(harmonics)
            stdata["annual_msl_detided"] = (tg_y - tide_fun(tg_y.index.values)).mean()

        if "annual_msl_detided_90d" not in stdata:
            tide_fun = u_gauge.harmonics_fun(harmonics_90d)
            stdata["annual_msl_detided_90d"] = (tg_y - tide_fun(tg_y.index.values)).mean()
    print(f"Processing {tg_src}: {df.shape[0]}/{df.shape[0]} ...")

    if gaugedata_discard_duplicates(data):
        changed = True

    return changed


def main():
    parser = argparse.ArgumentParser(description='Gather gauges within each flood map area.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    parser.add_argument('--offline', action="store_true",
                        help='If set, do not attempt to download new gauge data (esp. WSL).')
    args = parser.parse_args()

    source = args.source
    offline = args.offline
    print(source)

    meta_file = u_const.FLOODMAPS_DIR / source / "meta.hdf5"
    fm_meta = pd.read_hdf(meta_file).sort_values(by="date")

    # load tide gauge indices
    tg_df = {
        "codec": codec_meta_df(),
        "gtsm": pd.read_hdf(u_const.GSLCOMP_GTSM_INDEX_FILE),
        "gesla": pd.read_hdf(u_const.GSLCOMP_GESLA_INDEX_FILE),
        "wsl": wsl_meta_df(),
        "uhslc": pd.read_csv(UHSLC_DIR / "all_gauges.csv"),
        "gesla3": gesla3_meta_df(),
    }

    tg_gdf = {}
    for tg_src, df in tg_df.items():
        tg_gdf[tg_src] = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

    has_gauges = {}
    if "has_gauges" in fm_meta.columns:
        has_gauges = {row.map_id: row.has_gauges for _, row in fm_meta.iterrows()}
    set_has_gauges_any = False
    for _, row in fm_meta.iterrows():
        map_id = row['map_id']
        date = row['date']
        bounds = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        b_box = shapely.geometry.box(*bounds)
        date = np.datetime64(date)
        start, end = date - np.timedelta64(22, 'D'), date + np.timedelta64(1, 'D')

        outpath = u_const.GAUGES_DIR / source / "records" / f"{map_id}.pickle"
        data = {}
        if outpath.exists():
            with outpath.open("rb") as fp:
                data = pickle.load(fp)

        set_data_any = False
        for tg_src, gdf in tg_gdf.items():
            if tg_src not in data:
                data[tg_src] = []
                set_data_any = True

            args = (tg_src, data[tg_src], gdf, b_box, date, start, end)
            kwargs = dict(offline=offline) if tg_src == "wsl" else {}
            if gaugedata_from_date_and_box(*args, **kwargs):
                set_data_any = True

        if set_data_any:
            print(f"Writing to {outpath} ...")
            with outpath.open("wb") as fp:
                pickle.dump(data, fp)

        if map_id not in has_gauges:
            has_gauges[map_id] = sum(len(tgd) for tgd in data.values()) > 0
            set_has_gauges_any = True

    if set_has_gauges_any:
        fm_meta['has_gauges'] = fm_meta.map_id.apply(lambda i: has_gauges[i])
        print(f"Writing to {meta_file} ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            fm_meta.to_hdf(meta_file, "data")


if __name__ == "__main__":
    main()
