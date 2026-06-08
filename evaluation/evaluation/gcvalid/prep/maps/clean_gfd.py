"""
Rename GFD flood maps according to IBTrACS ID and reduce to values 0/1
"""
import zipfile

import numpy as np
import pandas as pd
import rasterio
import rasterio.io
import rasterio.merge

import gcvalid.util.constants as u_const


GFD_DATA_DIR = u_const.FLOODMAPS_DIR / "gfd"

GFD_INDEX_DIR = u_const.INDEX_DIR / "gfd"


def gen_gfd_map(zip_path, out_path):
    print(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        tifs = [f for f in zf.namelist() if f[-4:] == ".tif"]

    with zip_path.open("rb") as zfp, rasterio.io.ZipMemoryFile(zfp) as memf:
        if len(tifs) == 1:
            with memf.open(tifs[0]) as src:
                flood = src.read(1)
                transform = src.transform
        elif len(tifs) == 2:
            with memf.open(tifs[0]) as src1, memf.open(tifs[1]) as src2:
                flood, transform = rasterio.merge.merge([src1, src2], indexes=[1], method="max")
                flood = flood[0, :, :]
        else:
            raise TypeError("More than 2 tif-files in ZIP!")
    flood[flood > 0] = 1
    flood = flood.astype(np.uint8)

    kwargs = {
        "driver": "GTiff",
        "compress": "deflate",
        "height": flood.shape[0],
        "width": flood.shape[1],
        "count": 1,
        "dtype": np.uint8,
        "nodata": 255,
        "crs": u_const.DEFAULT_CRS,
        "transform": transform,
    }
    with rasterio.open(out_path, "w", **kwargs) as dst:
        dst.write(flood, 1)


def main():
    df = pd.read_csv(GFD_INDEX_DIR / "linked_ids.csv")
    for path in (GFD_DATA_DIR / "raw").glob("*.zip"):
        dfo_id = int(path.name.split("_")[1])
        ibtracs_id = df[df['ID'] == df]['ibtracs_id'].values[0]
        out_path = GFD_DATA_DIR / "clean_by_sid" / f"{ibtracs_id}-0.tif"
        if out_path.exists():
            continue
        gen_gfd_map(path, out_path)


if __name__ == "__main__":
    main()
