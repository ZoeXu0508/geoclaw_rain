"""
Download high water mark data from USGS flood event viewer (https://stn.wim.usgs.gov/fev/)

We went manually through all TC events for which we have flood maps and checked the availability
of high water marks for each of them. This script downloads the respective data and converts
values in feet to meters.
"""
import json

import numpy as np
import pyproj
import requests

import gcvalid.util.constants as u_const


FEV_DATA_FILE = u_const.HWMS_DIR / "raw.json"
"""Response of API request (see below)"""

FEV_API_REQUEST = (
    "https://stn.wim.usgs.gov/STNServices/HWMs/FilteredHWMs.json"
    "?Event=18,283,180,182,133,189,135,287"
    "&EventType=2"
    "&EventStatus=0"
    "&States="
    "&County="
    "&HWMType="
    "&HWMQuality="
    "&HWMEnvironment="
    "&SurveyComplete="
    "&StillWater="
)
"""API request generated on https://stn.wim.usgs.gov/STNDataPortal/"""

FEV_EVENTS = {
    "2003249N14329": "Isabel September 2003",  # https://stn.wim.usgs.gov/fev/#IsabelSeptember2003
    "2012234N16315": "Isaac Aug 2012",  # https://stn.wim.usgs.gov/fev/#IsaacAug2012
    "2016273N13300": "Matthew October 2016",  # https://stn.wim.usgs.gov/fev/#MatthewOctober2016
    "2017228N14314": "Harvey Aug 2017",  # https://stn.wim.usgs.gov/fev/#HarveyAug2017
    "2017242N16333": "Irma September 2017",  # https://stn.wim.usgs.gov/fev/#IrmaSeptember2017
    "2017260N12310": "Maria September 2017",  # https://stn.wim.usgs.gov/fev/#MariaSeptember2017
    "2018242N13343": "Florence Sep 2018",  # https://stn.wim.usgs.gov/fev/#FlorenceSep2018
    "2018280N18273": "Michael Oct 2018",  # https://stn.wim.usgs.gov/fev/#MichaelOct2018
}
"""List of selected TC events in FEV with corresponding IBTrACS IDs"""

FT_TO_M = 0.3048
"""Conversion factor from feet to meters"""

EPSG_CODES = {
    "NAD27": 4267,
    "NAD83": 4269,
    "NAD 83 (2011) epoch 2010": 6318,
    "NGVD29": 7968,
    "NAVD88": 5703,
    "PRVD02": 6641,
    "WGS84 (from Digital Map)": 9056,
}

CRS_WGS84_EGM96 = pyproj.crs.CRS.from_string("EPSG:9056+5773")

CACHED_PROJ_TRANSFORMS = {}


def get_proj_transform(s):
    if s not in CACHED_PROJ_TRANSFORMS:
        CACHED_PROJ_TRANSFORMS[s] = pyproj.transformer.Transformer.from_crs(
            crs_from=pyproj.crs.CRS.from_string(s),
            crs_to=CRS_WGS84_EGM96,
        )
    return CACHED_PROJ_TRANSFORMS[s]


def main():
    if not FEV_DATA_FILE.exists():
        r = requests.get(FEV_API_REQUEST)
        FEV_DATA_FILE.write_text(r.text)

    with FEV_DATA_FILE.open("r") as fp:
        data = json.load(fp)
    print(sorted(set([d['eventName'] for d in data])))
    print(sorted(set.union(*[set(d.keys()) for d in data])))

    print("Download file information ...")
    for i, d in enumerate(data):
        print(f"{100 * i / len(data):.1f}%", end='\r', flush=True)
        files_f = u_const.HWMS_DIR / "attachments" / f"files_{d['hwm_id']}.json"
        url = f"https://stn.wim.usgs.gov/STNServices/HWMs/{d['hwm_id']}/Files.json"
        if not files_f.exists():
            files_f.write_text(requests.get(url).text)
        with files_f.open("r") as fp:
            d['files'] = json.load(fp)
        # file name is in 'name', id is in 'file_id'
        # obtain data from https://stn.wim.usgs.gov//STNServices/Files/{file_id}/Item
    print()

    print("Convert vertical/horizontal datum to WGS84+EGM96 ...")
    data_out = []
    h_ignore = []
    v_ignore = []
    for i, d in enumerate(data):
        print(f"{100 * i / len(data):.1f}%", end='\r', flush=True)

        if d["horizontalDatumName"] not in EPSG_CODES:
            h_ignore.append(d["horizontalDatumName"])
            continue
        h_epsg = EPSG_CODES[d["horizontalDatumName"]]

        if d["verticalDatumName"] not in EPSG_CODES and "elev_ft" in d:
            v_ignore.append(d["verticalDatumName"])
            continue

        if "elev_ft" not in d:
            tform = get_proj_transform(f"EPSG:{h_epsg}")
            lat, lon, _ = tform.transform(d["latitude"], d["longitude"], 0)
            d["elev_m"] = np.nan
            d["latitude"] = lat
            d["longitude"] = lon
        else:
            v_epsg = EPSG_CODES[d["verticalDatumName"]]
            tform = get_proj_transform(f"EPSG:{h_epsg}+{v_epsg}")
            lat, lon, z = tform.transform(d["latitude"], d["longitude"], FT_TO_M * d['elev_ft'])
            d["elev_m"] = z
            d["latitude"] = lat
            d["longitude"] = lon

        d['height_above_gnd_m'] = (
            FT_TO_M * d['height_above_gnd'] if 'height_above_gnd' in d else np.nan
        )
        d['ibtracs_id'] = [k for k, val in FEV_EVENTS.items() if val == d['eventName']][0]

        data_out.append(d)
    print()

    print("Horizontal datums ignored:", ", ".join(np.unique(h_ignore)))
    print("Vertical datums ignored:", ", ".join(np.unique(v_ignore)))

    outpath = u_const.HWMS_FILE
    print(f"Writing to {outpath}...")
    with outpath.open("w") as fp:
        json.dump(data_out, fp)


if __name__ == "__main__":
    main()
