"""
Generate meta.hdf5 for flood maps
"""
import argparse
import warnings

import pandas as pd
import rasterio
import rasterio.warp

import gcvalid.util.constants as u_const


MAP_RAPID_IDS = (
    pd.read_csv(u_const.INDEX_DIR / "rapid" / "linked_ids.csv")[['rapid_id', 'ibtracs_id']]
)

DFO_DATES = pd.DataFrame([
    ("2001157N28265-0", "2001-06-11"),
    ("2003196N05150-0", "2003-07-25"),
    ("2003196N05150-1", "2003-07-26"),
    ("2003249N14329-0", "2003-09-20"),
    ("2003262N17254-0", "2003-09-24"),
    ("2004056S18125-0", "2004-03-07"),
    ("2004061S12072-0", "2004-03-11"),
    ("2004072S11146-0", "2004-04-01"),
    ("2004081S15148-0", "2004-03-23"),
    ("2004174N14146-0", "2004-07-13"),
    ("2004231N09147-0", "2004-08-30"),
    ("2004258N16300-0", "2004-10-04"),
    ("2004258N16300-1", "2004-09-21"),
    ("2004258N16300-2", "2004-09-21"),
    ("2004319N10134-0", "2004-11-24"),
    ("2005017S09061-0", "2005-02-02"),
    ("2005236N23285-0", "2005-09-02"),
    ("2005236N23285-1", "2005-09-09"),
    ("2005236N23285-2", "2005-09-07"),
    ("2005236N23285-3", "2005-09-02"),
    ("2005236N23285-4", "2005-09-04"),
    ("2005257N15120-0", "2005-09-22"),
    ("2005261N21290-0", "2005-09-26"),
    ("2005275N19274-0", "2005-10-12"),
    ("2005275N19274-1", "2005-10-12"),
    ("2005275N19274-2", "2005-10-12"),
    ("2005275N19274-3", "2005-10-12"),
    ("2005289N18282-0", "2005-10-25"),
    ("2005300N10279-0", "2005-11-15"),
    ("2006237N13298-0", "2006-09-02"),
    ("2006240N12265-0", "2006-09-04"),
    ("2007244N12303-0", "2007-09-06"),
    ("2007244N12303-1", "2007-09-09"),
    ("2007244N12303-2", "2007-09-06"),
    ("2007297N18300-0", "2007-11-04"),
    ("2007345N18298-0", "2007-12-16"),
    ("2008117N11090-0", "2008-05-05"),
    ("2011233N15301-0", "2011-09-01"),
    ("2012010S24049-0", "2012-01-29"),
    ("2016273N13300-0", "2016-10-05"),
    ("2016273N13300-1", "2016-10-25"),
    ("2017228N14314-0", "2017-09-08"),
    ("2017242N16333-0", "2017-09-16"),
    ("2017242N16333-1", "2017-09-16"),
    ("2017242N16333-2", "2017-09-18"),
    ("2017242N16333-3", "2017-09-18"),
    ("2017253N14130-0", "2017-09-16"),
    ("2017260N12310-0", "2017-09-21"),
    ("2017277N11279-0", "2017-10-07"),
    ("2017277N11279-1", "2017-10-06"),
    ("2017304N11127-0", "2017-11-07"),
    ("2018242N13343-0", "2018-09-20"),
    ("2018280N18273-0", "2018-10-11"),
    ("2018292N14261-0", "2018-10-26"),
    ("2019063S18038-0", "2019-03-23"),
    ("2019112S10053-0", "2019-04-28"),
    ("2019116N02090-0", "2019-05-07"),
    ("2019192N29274-0", "2019-07-15"),
    ("2019261N28264-0", "2019-09-22"),
    ("2019264N19071-0", "2019-09-25"),
    ("2019329N09160-0", "2019-12-06"),
], columns=("map_id", "date"))
"""Latest dates noted on the images or on the website that serves the image/tif"""

RAPID_DATES = pd.DataFrame([
    ("2016-10-02-2016-11-02-4601", "2016-10-23"),
    ("2017-08-25-2017-09-13-3424", "2017-09-05"),
    ("2017-09-06-2017-11-11-5796", "2017-09-18"),
    ("2018-09-11-2018-10-22-1625", "2018-09-19"),
    ("2018-10-10-2018-10-14-4255", "2018-10-13"),
    ("2019-07-10-2019-07-13-132", "2019-07-11"),
], columns=("rapid_id", "date"))
"""Latest dates mentioned in RAPID data set"""


def fm_raster(filename):
    with rasterio.open(filename, "r") as src:
        if src.crs == u_const.DEFAULT_CRS:
            [xmin, ymin, xmax, ymax] = src.bounds
            width, height = src.width, src.height
            xres, yres = src.transform[0], src.transform[4]
        else:
            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs, u_const.DEFAULT_CRS, src.width, src.height, *src.bounds)
            xmin, ymin = transform[2], transform[5]
            xmax, ymax = xmin + transform[0] * width, ymin + transform[4] * height
            if transform[0] < 0:
                xmin, xmax = xmax, xmin
            if transform[4] < 0:
                ymin, ymax = ymax, ymin
            xres, yres = transform[0], transform[4]
    return [width, height, xmin, ymin, xmax, ymax, xres, yres]


def gfd_date(dfo_ids):
    for dfo_id in dfo_ids.split(" and "):
        paths = list((u_const.FLOODMAPS_DIR / "gfd" / "raw").glob(f"*_{dfo_id}_*.zip"))
        if len(paths) > 0:
            date_str = paths[0].stem.split("_")[-1]
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    print(f"No GFD date found for {dfo_ids}")
    return ""


def map_dfo_ids():
    archive = pd.read_csv(u_const.INDEX_DIR / "dfo" / "archive_link_ibtracs.csv")
    archive = archive[~archive["ibtracs_id"].isna()]
    df_map = (
        archive[["ID", "ibtracs_id"]]
        .groupby("ibtracs_id")
        .apply(lambda group: " and ".join(group["ID"].astype(str).values))
    )
    df_map.name = "dfo_id"
    return df_map.reset_index()


def main():
    parser = argparse.ArgumentParser(description='Generate meta.hdf5 for flood maps.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    tif_path = u_const.FLOODMAPS_DIR / source / "clean_by_sid"
    df = pd.DataFrame(
        [[fname.stem] + fm_raster(fname) for fname in tif_path.glob("*.tif")],
        columns=["map_id", "width", "height", "xmin", "ymin", "xmax", "ymax", "xres", "yres"])
    df['ibtracs_id'] = df['map_id'].str.slice(0, 13)
    df = df.merge(map_dfo_ids(), left_on="ibtracs_id", right_on="ibtracs_id", how="left")
    df = df.merge(MAP_RAPID_IDS, left_on="ibtracs_id", right_on="ibtracs_id", how="left")

    if source == "gfd":
        df['date'] = df['dfo_id'].apply(gfd_date)
    elif source == "rapid":
        df = df.merge(RAPID_DATES, left_on='rapid_id', right_on='rapid_id', how='left')
    elif source == "dfo":
        df = df.merge(DFO_DATES, left_on='map_id', right_on='map_id', how='left')

    out_path = u_const.FLOODMAPS_DIR / source / "meta.hdf5"
    print(f"Writing to {out_path}...")
    with warnings.catch_warnings():
        # ignore warning about pickling of string columns
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        df.to_hdf(out_path, "data")


if __name__ == "__main__":
    main()
