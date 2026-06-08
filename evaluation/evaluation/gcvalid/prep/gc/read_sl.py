"""
Get the base sea level used in GeoClaw runs
"""
import argparse
import re

import numpy as np
import pandas as pd

import gcvalid.util.constants as u_const


def geoclaw_regions(path):
    regions = [re.sub(r" +", " ", l).split(" ") for l in path.read_text().split("\n")]
    regions = [(float(r[4]), float(r[6]), float(r[5]), float(r[7])) for r in regions
               if len(r) >= 8 and r[0] == "6"]
    return regions


def geoclaw_sea_level(path):
    return float([l.split(" ")[0] for l in path.read_text().split("\n") if "sea_level" in l][0])


def overlaps_any(bounds, regions):
    for r in regions:
        if (r[0] <= bounds[2] and bounds[0] <= r[2]
                and r[1] <= bounds[3] and bounds[1] <= r[3]):
            return True
    return False


def read_sea_level(source, map_id, fm_bounds):
    ibtracs_id = map_id.split("-")[0]
    glob = f"{ibtracs_id}_{source}-zos_*regions.data"
    sea_level = {}
    for path in (u_const.GEOCLAW_DIR / source / "meta").glob(glob):
        zos = path.name.split("-")[1].split("_")[1]
        regions = geoclaw_regions(path)
        if overlaps_any(fm_bounds, regions):
            path = path.parent / path.name.replace("regions.data", "geoclaw.data")
            if zos not in sea_level:
                sea_level[zos] = []
            sea_level[zos].append(geoclaw_sea_level(path))
    for zos, vals in sea_level.items():
        sea_level[zos] = np.amax(vals)
    sea_level['map_id'] = [map_id]
    return pd.DataFrame(sea_level)


def main():
    parser = argparse.ArgumentParser(description='Get the base sea level used in GeoClaw runs.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5").set_index("map_id")
    data = []
    for fm_fname in (u_const.FLOODMAPS_DIR / source / "clean_by_sid").glob("*.tif"):
        map_id = fm_fname.stem
        meta_row = meta.loc[map_id]
        fm_bounds = tuple(meta_row[['xmin', 'ymin', 'xmax', 'ymax']])
        data.append(read_sea_level(source, map_id, fm_bounds))
    df = pd.concat(data).sort_values(by="map_id").fillna(0)
    df.to_csv(u_const.GEOCLAW_DIR / source / "base_sea_level.csv", index=False)

if __name__ == "__main__":
    main()
