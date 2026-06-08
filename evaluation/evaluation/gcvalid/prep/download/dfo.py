"""
Download DFO flood archive and TC-related flood maps

Most of the flood maps available from DFO are listed in a flood map index, but the website provides
additional flood maps that are not listed in that index. We went manually through the pages to
identify additional sources of TC-related flood maps.

- We do not take files into consideration that lack meta data such as provided in the
[GISdata](http://floodobservatory.colorado.edu/GISdata/) and
[Version3](http://floodobservatory.colorado.edu/Version3/) subdirectories of the web site. This
also applies to the "Large Region Displays of Surface Water Records" listed on the
[Recent.html](http://floodobservatory.colorado.edu/Recent.html) page.
"""
import dateutil.parser
import re
import sys

import numpy as np
import pandas as pd
from pyquery import PyQuery as pq
import requests
import xarray as xr

import gcvalid.util.constants as u_const

IBTRACS_FILE = u_const.TRACKS_DIR / "raw" / "IBTrACS.ALL.v04r00.nc"

DFO_DATA_DIR = u_const.FLOODMAPS_DIR / "dfo"

DFO_INDEX_DIR = u_const.INDEX_DIR / "dfo"

FLOOD_ARCHIVE_X_FILE = DFO_INDEX_DIR / "FloodArchive.xlsx"

MAP_INDEX_HTML_FILE = DFO_INDEX_DIR / "MapIndex.htm"

MAP_INDEX_CSV_FILE = DFO_INDEX_DIR / "MapIndex.csv"

ARCHIVE_MAPLINKED_FILE = DFO_INDEX_DIR / "archive_link_mapindex.csv"

ARCHIVE_TCLINKED_FILE = DFO_INDEX_DIR / "archive_link_ibtracs.csv"

MAP_INDEX_URL = "http://floodobservatory.colorado.edu/FloodMapIndex.htm"
"""Index of events in the DFO flood archive for which flood maps are available

Note that this index contains all events listed in the "Master Index 2000 to 2008":
http://floodobservatory.colorado.edu/Archives/MapIndex.htm

It contains 386 events between 2000 to 2008 without machine readable (e.g. shp or gtiff) maps, but
with various types of RGB image formats (JPEG, PNG, GIF). There are no events for the years 2009
through 2013. Starting from 2014, there are 63 events that come together with GeoTIFF flood maps.
"""

FLOOD_ARCHIVE_X_URL = "https://floodobservatory.colorado.edu/temp/FloodArchive.xlsx"
"""DFO archive of inland and coastal floods (not related to the availability flood maps)

Formerly: http://floodobservatory.colorado.edu/Version3/FloodArchive.xlsx

For this study, the most interesting information in this file is about the "main cause".
"""

RR_DIR_URL = "http://floodobservatory.colorado.edu/RapidResponse/"
RR_DIR_TCS = [
    {"ID": 3861, "url": RR_DIR_URL + "2011SouthNewJerseyiRR.html"},
    {"ID": 3892, "url": RR_DIR_URL + "2012MozambiqueRR.html"},
    {"ID": 3997, "url": RR_DIR_URL + "Sandy.html"},
]
"""URLs to TC-related rapid response maps

Some of the flood maps in the RapidResponse directory are listed on
the [Recent.html](http://floodobservatory.colorado.edu/Recent.html) page.
But we were able to locate flood maps for additional TC-related events manually.
"""

MANUAL_IBTRACS_BY_DFO_ID = {
    1587: '2000032S11116',
    1590: '2000056S17152',
    1660: '2000305N06136',
    1733: '2001157N28265',
    1922: '2002122S07063',
    2254: '2003158N08156',
    2284: '2003196N05150',
    2291: '2003196N05150',
    2352: '2003249N14329',
    2356: '2003262N17254',
    2444: '2004056S18125',
    2456: '2004061S12072',
    2458: '2004081S15148',
    2461: '2004072S11146',
    2516: '2004174N14146',
    2554: '2004231N09147',
    2560: '2004238N11325',
    2566: '2004258N16300',
    2587: '2004319N10134',
    2619: '2005017S09061',
    2719: '2005236N23285',
    2732: '2005257N15120',
    2735: '2005261N21290',
    2744: '2005275N19274',
    2753: '2005289N18282',
    2759: '2005300N10279',
    2959: '2006237N13298',
    2960: '2006240N12265',
    3184: '2007244N12303',
    3218: '2007297N18300',
    3239: '2007345N18298',
    3302: '2008117N11090',
    3316: "2008152N18273",
    3742: "2010302N09306",
    3892: "2012010S24049",
    3963: "2012209N11131",
    4100: "2013306N07162",
    4202: '2014285N16305',
    4217: '2014362N07130',
    4224: "2015013S18038",
    4282: '2015207N22091',
    4523: "2017260N12310",
    4643: "2018179N19134",
    4674: '2018227N11145',
    4745: '2019112S10053',
    4771: '2019192N29274',
    4802: '2019264N19071',
    4827: '2019329N09160',
    4964: '2020254N25287',
    5006: "2020258N18332",
    5022: "2021029S16171",
}
"""Manual matchings of DFO to ibtracs_id IDs (where the automatic matching is ambiguous)"""


FLOOD_MAP_URLS = [
    "https://floodobservatory.colorado.edu/images/2001-051Radarsat.jpg",
    "https://floodobservatory.colorado.edu/images/2003170Luzon.jpg",
    "https://floodobservatory.colorado.edu/images/2003177Guangxi.jpg",
    "https://floodobservatory.colorado.edu/images/2003239Isabel.jpg",
    "https://floodobservatory.colorado.edu/images/2003243Mexico.jpg",
    "https://floodobservatory.colorado.edu/images/2004033Pilbara.jpg",
    "https://floodobservatory.colorado.edu/images/2004045NMad.jpg",
    "https://floodobservatory.colorado.edu/images/2004050DeGrey.jpg",
    "https://floodobservatory.colorado.edu/images/2004047NoQld.jpg",
    "https://floodobservatory.colorado.edu/images/2004105Taiwan.jpg",
    "https://floodobservatory.colorado.edu/images/2004143Luzon.jpg",
    "https://floodobservatory.colorado.edu/images/2004155Forida.jpg",
    "https://floodobservatory.colorado.edu/images/2004155Haiti.jpg",
    "https://floodobservatory.colorado.edu/images/2004155DomRep.jpg",
    "https://floodobservatory.colorado.edu/images/2004176Phil.jpg",
    "https://floodobservatory.colorado.edu/images/2005014SMadagascar.jpg",
    "https://floodobservatory.colorado.edu/images/2005114Biloxi.jpg",
    "https://floodobservatory.colorado.edu/images/2005114KatrinaSFla.jpg",
    "https://floodobservatory.colorado.edu/images/2005114NewOrleans.jpg",
    "https://floodobservatory.colorado.edu/images/2005114Mobile.jpg",
    "https://floodobservatory.colorado.edu/images/2005114MissDelta.jpg",
    "https://floodobservatory.colorado.edu/images/2005127NgheAn.jpg",
    "https://floodobservatory.colorado.edu/images/2005130GulfCoastRita.jpg",
    "https://floodobservatory.colorado.edu/images/2005139SMexico.jpg",
    "https://floodobservatory.colorado.edu/images/2005139Retalhuleu.jpg",
    "https://floodobservatory.colorado.edu/images/2005139Escuintla.jpg",
    "https://floodobservatory.colorado.edu/images/2005139CenAmer.jpg",
    "https://floodobservatory.colorado.edu/images/2005148WilmaSFla.jpg",
    "https://floodobservatory.colorado.edu/images/2005154HondNic.jpg",
    "https://floodobservatory.colorado.edu/images/2006183NeCapeFear.jpg",
    "https://floodobservatory.colorado.edu/images/2006184SBaja.jpg",
    "https://floodobservatory.colorado.edu/images/2007177MexPanuco.jpg",
    "https://floodobservatory.colorado.edu/images/2007177HondMotagua.jpg",
    "https://floodobservatory.colorado.edu/images/2007177NicMiskito.jpg",
    "https://floodobservatory.colorado.edu/images/2007211DomRep.jpg",
    "https://floodobservatory.colorado.edu/images/2007232Yaque.jpg",
    "https://floodobservatory.colorado.edu/images/2008052Burma.jpg",
    "https://floodobservatory.colorado.edu/RapidResponse/SNewJersey.tif",
    "https://floodobservatory.colorado.edu/RapidResponse/2012Mozambique.tif",
    "https://floodobservatory.colorado.edu/Events/2016USA4402/20161005USA4402HaitiGrandAnseSUDALOS.tif",
    "https://floodobservatory.colorado.edu/Events/2016USA4402/2016USA4402NC.tif",
    "https://floodobservatory.colorado.edu/Events/2017USA4510/2017USA4510.tif",
    "https://floodobservatory.colorado.edu/Events/2017USA4516/2017USA4516CentralFlorida.tif",
    "https://floodobservatory.colorado.edu/Events/2017USA4516/2017USA4516SouthFlorida.tif",
    "https://floodobservatory.colorado.edu/Events/2017USA4516/2017USA4516Cuba.tif",
    "https://floodobservatory.colorado.edu/Events/2017USA4516/2017USA4516Haiti.tif",
    "https://floodobservatory.colorado.edu/Events/2017Vietnam4518/2017Vietnam4518a.tif",
    "https://floodobservatory.colorado.edu/Events/2017Vietnam4518/2017Vietnam4518b.tif",
    "https://floodobservatory.colorado.edu/Events/2017USA4523/2017USA4523PuertoRico.tif",
    "https://floodobservatory.colorado.edu/Events/2017USA4524/2017USA4524EAST.tif",
    "https://floodobservatory.colorado.edu/Events/2017USA4524/2017USA4524West.tif",
    "https://floodobservatory.colorado.edu/Events/2017Vietnam4533/2017Vietnam4533.tif",
    "https://floodobservatory.colorado.edu/Events/4676/2018USA4676.tif",
    "https://floodobservatory.colorado.edu/Events/4687/2018USA4687.tif",
    "https://floodobservatory.colorado.edu/Events/4695/2018Mexico4695Willa.tif",
    "https://floodobservatory.colorado.edu/Events/4725/2019Malawi4725CombinedLarge.tif",
    "https://floodobservatory.colorado.edu/Events/4745/2019Mozambique4745.jpg",
    "https://floodobservatory.colorado.edu/Events/4746/2019India4746.jpg",
    "https://floodobservatory.colorado.edu/Events/4771/4771.tif",
    "https://floodobservatory.colorado.edu/Events/4797/2019USA4797.tif",
    "https://floodobservatory.colorado.edu/Events/4802/2019Oman4802.tif",
    "https://floodobservatory.colorado.edu/Events/4827/2019Philippines4827.tif",
]
"""List of flood map files to use for this study

The list is the result of going manually through all TC-related flood maps and checking for usable
data, image and/or file formats and skipping all flood maps that do not contain any coastal areas.
"""


def map_index_to_csv():
    """Convert DFO's HTML MapIndex to CSV"""
    if not MAP_INDEX_HTML_FILE.exists():
        r = requests.get(MAP_INDEX_URL)
        MAP_INDEX_HTML_FILE.write_text(r.text)

    with MAP_INDEX_HTML_FILE.open("r", encoding="windows-1252") as fp:
        html = fp.read()

    html = pq(html)("table").eq(0).html()
    html = re.sub(r"\xa0", " ", html)
    html = re.sub(r"( *&nbsp; *)+", " ", html)
    html = re.sub(r" *(colspan|span|align|style|class|height|width|x:[^=]+)=\"[^\"']*\"", "", html)
    html = re.sub(r"</?(col|span|font|br) ?/?>", "", html)
    html = re.sub(r" *\n *", "\n", html)
    html = re.sub(r">\n<", "><", html)
    html = re.sub(r" *\n *", " ", html)
    html = re.sub(r", *;", ",", html)
    html = re.sub(r"<td>[  ]*</td>", "<td></td>", html)
    html = re.sub(r"<td ?/>", "<td></td>", html)
    html = re.sub(r"</tr>", "\n", html)
    html = re.sub(r"</td> *<td>", "|", html)
    html = re.sub(r"</?t[rd]/?>", "", html)
    html = re.sub(r"(; +)+", "; ", html)
    html = re.sub(r"^[; ]*$", "", html, flags=re.M)
    html = re.sub(r"\n+", "\n", html)
    html = re.sub(r"^\n+", "", html)
    html = re.sub(r" *\| *", "|", html)
    html = re.sub(
        r"<a href=\"https?://[a-z\.]+(/~floods)?/index\.html\"[^>]*>([^<]+)</a>", r"\2", html,
    )
    html = re.sub(r"\n(\|*\n)+", "\n", html)
    html = re.sub(r"\n([^\|]*\n)+", "\n", html)

    dataset = []
    for line in html.split("\n")[1:]:
        if "|" not in line or ("floodobservatory.colorado.edu" not in line
                               and "www.dartmouth.edu" not in line):
            continue
        url = [m.group(1) for m in re.finditer(r"href=\"([^\"]+)\"", line)]
        assert len(url) == 1
        url = url[-1]
        line = re.sub(r"<a href=\"[^\"]+\"[^>]*>([^<]+)</a>", r"\1", line)
        fields = line.split("|")[:4]
        if fields[0] == '#N/A' and fields[2] == 'South Africa' and fields[3] == '3/13/00':
            # Manually correcting, assuming we have the right match ...
            fields[0] = '1593'
            fields[3] = '3/9/00'
        try:
            fields[0] = int(fields[0])
        except ValueError:
            if "Link to Previous Listing" not in fields[0]:
                print("Invalid DFO ID:", fields)
            continue
        fields.insert(2, url)
        if fields[-1] != "":
            date = dateutil.parser.parse(fields[-1])
            fields[-1] = date.strftime("%Y-%m-%d")
            if fields[1] != "":
                fields[1] = f"{date.year}-{int(fields[1]):03d}"
        dataset.append(fields)

    df = pd.DataFrame(data=dataset, columns=['dfo_id', 'old_dfo_id', 'url', 'countries', 'Began'])
    df.to_csv(MAP_INDEX_CSV_FILE, index=False)


def load_map_index():
    """Load DFO's HTML MapIndex as DataFrame"""
    if not MAP_INDEX_CSV_FILE.exists():
        map_index_to_csv()
    return pd.read_csv(MAP_INDEX_CSV_FILE)


def load_flood_archive():
    if not FLOOD_ARCHIVE_X_FILE.exists():
        r = requests.get(FLOOD_ARCHIVE_X_URL)
        with FLOOD_ARCHIVE_X_FILE.open("wb") as fp:
            fp.write(r.content)

    df = pd.read_excel(FLOOD_ARCHIVE_X_FILE).dropna(axis=0, subset=['ID'])
    df['OtherCountry'] = df['OtherCountry'].fillna("").astype(str).replace("0", "")
    df["MainCause"] = df["MainCause"].astype(str).str.replace("nan", "")
    df = df.groupby("ID").first().reset_index()
    return df


def get_matching_record(archive, mapi):
    candidates = archive[archive['Began'] == mapi['Began']]
    msk = np.array([c.replace("\xA0", " ") in mapi['countries']
                    or c.replace('USA', 'US') in mapi['countries']
                    or c.replace('South Korea', 'South and North Korea') in mapi['countries']
                    or c in mapi['countries'].replace("Caucasus", "Georgia")
                    or c in mapi['countries'].replace("Caribbean", "Haiti")
                    or c in mapi['countries'].replace("South Asia", "Thailand")
                    for c in candidates['Country']])
    if not any(msk):
        print("No matching record for entry:")
        if candidates.size > 0:
            print("'%s' not in '%s'" % (candidates.iloc[0]['Country'], mapi['countries']))
            print(["\\x%x" % ord(c) for c in candidates.iloc[0]['Country']])
            print(["\\x%x" % ord(c) for c in mapi['countries']])
        print(mapi)
        print(candidates)
        sys.exit()
    if "US-Midwest" == mapi['countries']:
        msk &= [-110 <= l <= -80 for l in candidates['long']]
    if "2003-053" == mapi['old_dfo_id']:
        msk &= [l < 35 for l in candidates['lat']]
    if any(i in mapi['old_dfo_id'] for i in ["2003-079", "2006-174"]):
        msk &= [l != "" and l in mapi['countries'] for l in candidates['OtherCountry']]
    assert np.count_nonzero(msk) == 1
    return candidates[msk].iloc[0]['ID']


def link_mapindex_to_archive(archive):
    if ARCHIVE_MAPLINKED_FILE.exists():
        archive = pd.read_csv(ARCHIVE_MAPLINKED_FILE, parse_dates=["Began"])
        archive["MainCause"] = archive["MainCause"].fillna("").astype(str)
        archive["url"] = archive["url"].fillna("").astype(str)
        return archive

    mapindex = load_map_index()
    mapindex = mapindex.rename(columns={"dfo_id": "ID"})
    mapindex['Began'] = pd.to_datetime(mapindex['Began'])
    mapindex['Corr. ID'] = mapindex['ID']

    # manually correct some of the wrong IDs
    mapindex.loc[mapindex['ID'] == 4605, 'Corr. ID'] = 4606
    mapindex.loc[mapindex['ID'] == 3185, 'Corr. ID'] = 3184
    mapindex['ID'] = mapindex['Corr. ID']

    # deal with duplicate IDs in MapIndex
    dups = mapindex[mapindex.duplicated(subset=["ID"], keep=False)]
    for idx, row in dups.iterrows():
        mapindex.loc[idx, 'Corr. ID'] = get_matching_record(archive, row)
    mapindex['ID'] = mapindex['Corr. ID']

    # deal with typos in ID column of MapIndex (recognized by mismatch in "Began" record)
    typos = mapindex.merge(archive, on=["ID"], how="left", suffixes=["", "_y"])
    typos = typos[(typos['Began'] != typos['Began_y']) & ~typos['Began'].isna()]
    for idx, row in typos.iterrows():
        # for some, "Began" is wrong even though the rest is okay
        if row['ID'] not in [4178, 4201, 4337, 4474, 4523, 4606, 4795, 4796]:
            mapindex.loc[idx, 'Corr. ID'] = get_matching_record(archive, row)
    mapindex['ID'] = mapindex['Corr. ID']

    mapindex = mapindex.drop(columns=['Corr. ID']).sort_values(by=['ID'])
    assert not any(mapindex.duplicated(subset=["ID"], keep=False))

    archive = archive.merge(mapindex.drop(columns="Began"), on=["ID"], how="left")
    archive["url"] = archive["url"].fillna("").astype(str)
    archive.to_csv(ARCHIVE_MAPLINKED_FILE, index=False)
    return archive


def ibtracs_names_for_year(ibtracs_ds, year):
    ds_year = ibtracs_ds.sel(storm=(
        np.abs(ibtracs_ds.sid.str.slice(0, 4).astype(int) - year) <= 2
    ))
    sids = ds_year["sid"].astype(str).values
    names = ds_year["name"].astype(str).str.lower().values
    return xr.DataArray(names, coords={"sid": sids}, name="name")


def sids_for_year(df, year, ibtracs_ds):
    df_year = df[df["Began"].dt.year == year].copy()
    causes = df_year["MainCause"].str.lower().to_xarray()
    da_names = ibtracs_names_for_year(ibtracs_ds, year)
    matches = causes.str.contains(da_names, regex=False)
    match_idx = xr.Dataset({
            d: ("i", matches[d].values[i])
            for d, i in zip(matches.dims, matches.values.nonzero())
    }).to_dataframe().groupby("ID", group_keys=False)["sid"].apply(list).to_dict()

    # deal with misleading matches
    misleading_matches = [
        ("rai", "rain"),
        ("ele", "release"),
        ("tia", "torrential"),
        ("earl", "early"),
        ("ian", "pian"),
    ]
    for dfo_id in list(match_idx.keys()):
        ibtracs_ids = match_idx[dfo_id]
        ev_names = da_names.sel(sid=ibtracs_ids).values
        ev_cause = causes.sel(ID=dfo_id).item()
        for n, s in misleading_matches:
            if n in ev_names and n not in ev_cause.replace(s, ""):
                match_idx[dfo_id] = np.array(ibtracs_ids)[ev_names != n].tolist()
    match_idx = {k: v for k, v in match_idx.items() if len(v) > 0}

    # deal with non-unique matches
    for dfo_id in list(match_idx.keys()):
        ibtracs_ids = match_idx[dfo_id]
        if len(ibtracs_ids) <= 1:
            continue
        ev_names = da_names.sel(sid=ibtracs_ids).values
        ev_cause = causes.sel(ID=dfo_id).item()
        if ev_names[1] in ev_names[0]:
            ibtracs_ids = ibtracs_ids[::-1]
            ev_names = da_names.sel(sid=ibtracs_ids).values
        if len(ibtracs_ids) > 2 or ev_names[0] not in ev_names[1] or ev_names[0] == ev_names[1]:
            ev_began = df.loc[dfo_id, "Began"].values[0]
            print(
                f"Several TCs for one DFO entry ({dfo_id}):",
                ev_began, ev_cause, ev_names, ibtracs_ids,
            )
        else:
            match_idx[dfo_id] = [ibtracs_ids[1]]
    return {k: ",".join(v) for k, v in match_idx.items()}


def link_ibtracs_to_archive(archive):
    if ARCHIVE_TCLINKED_FILE.exists():
        archive = pd.read_csv(ARCHIVE_TCLINKED_FILE)
        archive["MainCause"] = archive["MainCause"].fillna("").astype(str)
        archive["url"] = archive["url"].fillna("").astype(str)
        archive["ibtracs_id"] = archive["ibtracs_id"].fillna("").astype(str)
        return archive

    archive = archive.set_index("ID")
    archive["ibtracs_id"] = pd.Series(MANUAL_IBTRACS_BY_DFO_ID)
    archive["ibtracs_id"] = archive["ibtracs_id"].fillna("")

    causes = archive["MainCause"].str.lower()
    archive["tc_related"] = (
        causes.str.contains("tropical")
        | causes.str.contains("tropial")
        | causes.str.contains("cyclone")
        | causes.str.contains("hurricane")
        | causes.str.contains("typhoon")
        | causes.str.contains("surge")
        | causes.str.contains("zorba")
    ) & (
        ((archive["lat"] > 0) & ~archive["Began"].dt.month.isin([1, 2, 3, 4]))
        | ((archive["lat"] < 0) & ~archive["Began"].dt.month.isin([7, 8, 9, 10]))
    )
    archive["tc_unrelated"] = ~archive["tc_related"] & (
        causes.str.contains("avalanche")
        | causes.str.contains("slide")
    )

    print("Reading IBTrACS NetCDF...")
    ibtracs_ds = xr.open_dataset(IBTRACS_FILE)

    msk = ~archive["tc_unrelated"] & (archive["ibtracs_id"] == "")
    _archive = archive[msk].copy()
    years = _archive["Began"].dt.year.values
    for y in range(years.min(), years.max() + 1):
        ibtracs_by_dfo_id = sids_for_year(_archive, y, ibtracs_ds)
        for dfo_id, sid in ibtracs_by_dfo_id.items():
            archive.loc[dfo_id, "ibtracs_id"] = sid
    archive.loc[~archive["tc_related"] & (archive["ibtracs_id"] != ""), "tc_related"] = True
    archive = archive.reset_index()
    archive.to_csv(ARCHIVE_TCLINKED_FILE, index=False)
    return archive


def get_subdir(base_name):
    subdir_URL = f"http://floodobservatory.colorado.edu/{base_name}/"
    subdir_html = DFO_INDEX_DIR / f"subdir_{base_name}.html"
    subdir_csv = DFO_INDEX_DIR / f"subdir_{base_name}.csv"

    if subdir_csv.exists():
        return pd.read_csv(subdir_csv)

    if not subdir_html.exists():
        r = requests.get(subdir_URL)
        subdir_html.write_text(r.text)
    html = subdir_html.read_text()

    files = []
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
        name = m.group(1).strip("/")
        if name in ["", ".DS_Store"]:
            continue
        files.append(name)
    df = pd.DataFrame({"name": files})
    df["url"] = (subdir_URL + df["name"] + "/").str.replace(
        r"\.(jpg|png|pdf|html)/", r".\1", case=False, regex=True,
    )
    df.to_csv(subdir_csv, index=False)
    return df


def scan_addtl_subdirs(archive):
    msk_maps = (archive["url"] != "")
    min_dfo_id = archive.loc[msk_maps, "ID"].min()
    archive = archive[archive["ID"] > min_dfo_id]

    dfs = [archive, pd.DataFrame(RR_DIR_TCS)]
    for subdir in ["Events", "images"]:
        df_subdir = get_subdir(subdir)
        subdir_tcs = []
        for _, tc_row in archive[archive["tc_related"]].iterrows():
            dfo_id =  tc_row["ID"]
            old_dfo_id =  tc_row["old_dfo_id"]
            for key, i in {"ID": dfo_id, "old_dfo_id": old_dfo_id}.items():
                msk = (
                    df_subdir["name"].str.contains(str(i))
                    | df_subdir["name"].str.contains(str(i).replace("-", ""))
                ) & ~np.any([
                    [f"/{subdir}/{name}" in row["url"] for name in df_subdir["name"].values]
                    for _, row in archive[archive[key] == i].iterrows()
                ], axis=0)
                subdir_tcs.extend([
                    {"ID": dfo_id, "old_dfo_id": old_dfo_id, "url": row["url"]}
                    for _, row in df_subdir[msk].iterrows()
                ])
        dfs.append(pd.DataFrame(subdir_tcs))
    archive = pd.concat(dfs).sort_values(by="ID").reset_index(drop=True)
    for col in ["MainCause", "Began", "ibtracs_id", 'tc_related']:
        archive[col] = archive.groupby("ID")[col].transform("first")
    archive["url"] = archive["url"].str.replace("http:", "https:")
    archive = archive.drop_duplicates(subset=["ID", "url"])

    tc_maps = (
        archive[(archive["url"] != "") & (archive["ibtracs_id"] != "")]
    ).reset_index(drop=True)

    print("These entries have been checked manually one by one for coastal flood maps:")
    print(tc_maps[["ID", "ibtracs_id", "url"]])


def download_flood_map_files():
    for url in FLOOD_MAP_URLS:
        fname = url.split("/")[-1]
        is_tif = fname.lower().endswith(".tif")
        path = DFO_DATA_DIR / ("geotiff" if is_tif else "images") / fname
        if not path.exists():
            print(f"Downloading from {url} ...")
            r = requests.get(url)
            with path.open("wb") as fp:
                fp.write(r.content)


def main():
    archive = load_flood_archive()
    archive = link_mapindex_to_archive(archive)
    archive = link_ibtracs_to_archive(archive)
    scan_addtl_subdirs(archive)
    download_flood_map_files()


if __name__ == "__main__":
    main()
