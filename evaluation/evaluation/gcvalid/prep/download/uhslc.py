"""
Download UHSLC database and extract list of tide gauge stations (and meta data)
"""
import pathlib
import re
import urllib.request
import zipfile

import pandas as pd

import gcvalid.util.constants as u_const


UHSLC_DIR = u_const.GAUGES_DIR / "uhslc_rqds"

UHSLC_DATA_FILE = UHSLC_DIR / "global.zip"

UHSLC_FTP_URL = "ftp://ftp.soest.hawaii.edu/uhslc/rqds/global.zip"


def read_gauge_data_file(zfp, fname):
    with zfp.open(fname) as fp:
        line = fp.read(80).decode().strip()
    m = re.match(
        r'^(.{4}) (.{18}) (.{19}) ([0-9]{4}) ([0-9 ]{5}[NS]) ([0-9 ]{6}[WES]) (.*)$',
        line,
    )
    if m is None:
        raise ValueError(f"Unknown header format: {line}")
    station_id = m.group(1).strip()
    name = m.group(2).strip()
    country = m.group(3).strip()
    year = int(m.group(4))
    s_lat = m.group(5).replace(" ", "0")
    lat = (1 if s_lat[-1] == "N" else -1) * (float(s_lat[:2]) + float(s_lat[2:5]) / 600)
    s_lon = m.group(6).replace(" ", "0")
    lon = float(s_lon[:-4]) + float(s_lon[-4:-1]) / 600
    if s_lon[-1] == "W":
        lon *= -1
    elif s_lon[-1] == "S":
        if station_id not in ["372B"]:
            raise ValueError(f"Unknown longitude format: {line}")
    return station_id, name, country, lat, lon, year


def read_gauge_zip(path, fp):
    basin = path.parent.parent.name
    with zipfile.ZipFile(fp) as zfp:
        meta = [read_gauge_data_file(zfp, fname) for fname in zfp.namelist()]
    if not all(m[0] == meta[0][0] for m in meta):
        print(sorted(set(m[0] for m in meta)))
        print("Inconsistent ID for station:", path)
    if not all(m[2] == meta[0][2] for m in meta):
        countries = sorted(set(m[2] for m in meta))
        if countries != ['Fd St Micronesia', 'Fd. St. Micronesia']:
            print(countries)
            print("Inconsistent country for station:", path)
    if not all(m[1] == meta[0][1] for m in meta):
        names = sorted(set(m[1] for m in meta))
        print(f"Inconsistent name for station {meta[0][0]} in {basin}: {names}")
    d = {"location_changed": any(m[3:5] != meta[0][3:5] for m in meta)}
    d["station_id"], d["name"], d["country"], d["lat"], d["lon"] = meta[-1][:-1]
    d["y_start"] = min(m[-1] for m in meta)
    d["y_end"] = max(m[-1] for m in meta)
    return d


def main():
    if not UHSLC_DATA_FILE.exists():
        urllib.request.urlretrieve(UHSLC_FTP_URL, filename=UHSLC_DATA_FILE)
    df = []
    with zipfile.ZipFile(UHSLC_DATA_FILE) as zfp:
        for zname in zfp.namelist():
            path = pathlib.Path(zname)
            if path.parent.name != "hourly" or not path.name.endswith(".zip"):
                continue
            with zfp.open(zname) as fp:
                df.append(read_gauge_zip(path, fp))
    df = pd.DataFrame(df)
    df = df.sort_values(by="station_id").reset_index(drop=True)
    df.to_csv(UHSLC_DIR / "all_gauges.csv", index=False)


if __name__ == "__main__":
    main()
