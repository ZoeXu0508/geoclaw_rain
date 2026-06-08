"""
Reproject CaMa-Flood data to flood map areas
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import xarray as xr

import gcvalid.util as util
import gcvalid.util.constants as u_const
import gcvalid.util.io as u_io


PLUVIAL_RAW_DIR = u_const.PLUVIAL_DIR / "raw"

CAMA_PROTECTION = {
    "no": "annual_max",
    "2y": "2y",
    "flopros": "flopros",
}

CAMA = {
    "isimip3a": {
         "path": PLUVIAL_RAW_DIR / "isimip3a",
         "fname": (
             "*/gswp3-w5e5/obsclim_histsoc_default/downscaling/"
             "cama*_gswp3-w5e5_obsclim_histsoc_default_flddph_{prot}"
             "_1arcmin_global_{y}.nc"
         ),
         "res": 60 / 3600,
         "var": "flddph",
    },

    # Old simulations, used to be in
    #
    # /p/projects/cama-flood/volkholz/cama-flood-runs/old_isimip3a_with_spurious_climate/ ...
    # ... /isimip3a_0prot_30arcsec/out/hydropy/gswp3-w5e5_obsclim/nat_default/ ...
    # ... /area-30arcsec-fit-obsclim_histsoc/fldfrc_annual_max_gev_0.1mmpd_protection-0.nc
    #
    # "isimip3a": {
    #     "path": PLUVIAL_RAW_DIR / "isimip3a",
    #     "fname": "fldfrc_annual_max_gev_0.1mmpd_protection-{prot}.nc",
    #     "res": 30 / 3600,
    #     "var": "fldfrc",
    # },

    # Old simulations no longer available on the cluster:
    #
    # "isimip2a": {
    #     "path":  PLUVIAL_RAW_DIR / "isimip2a" / "princeton/lpjml/area-150arcsec",
    #     "fname": "fldfrc_annual_max_gev_0.1mmpd_protection-{prot}.nc",
    #     "res": 150 / 3600,
    # },
}


def load_fm_transform(source, map_id):
    path = u_const.FLOODMAPS_DIR / source / "clean_by_sid" / f"{map_id}.tif"
    with rasterio.open(path, "r") as src:
        return src.width, src.height, src.transform, src.crs


def process_map(source, prot, meta):
    map_id = meta["map_id"]
    bounds = tuple(meta[['xmin', 'ymin', 'xmax', 'ymax']])

    fm_width, fm_height, fm_transform, fm_crs = load_fm_transform(source, map_id)

    date = np.datetime64(meta['date']).astype("datetime64[Y]")
    _, _, year = util.dt64_to_dmy(date)

    for key in CAMA.keys():
        prot_suffix = "flopros" if prot == "flopros" else f"{prot}prot"
        out_path = u_const.PLUVIAL_MAPS_DIR / f"{key}{prot_suffix}" / source / f"{map_id}.tif"
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if "*" not in CAMA[key]['fname']:
            cama_data, cama_transform = u_io.read_nc_with_date_and_bounds(
                CAMA[key]['path'] / CAMA[key]['fname'], CAMA[key]['var'], date, bounds)
        else:
            # compute multi-model median across all hydrological models in ISIMIP3
            paths = list(CAMA[key]['path'].glob(CAMA[key]['fname'].format(
                y=year, prot=CAMA_PROTECTION[prot],
            )))
            ds = (
                xr.open_mfdataset(paths, combine="nested", concat_dim=["model"])
                .median(dim="model", skipna=True)
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    module="numpy",
                    category=RuntimeWarning,
                )
                cama_data, cama_transform = u_io.read_ds_with_date_and_bounds(
                    ds, CAMA[key]['var'], date, bounds)
        cama_data[np.isnan(cama_data)] = 0

        out_kwargs = {
            "driver": "GTiff",
            "compress": "deflate",
            "height": fm_height,
            "width": fm_width,
            "count": 1,
            "dtype": cama_data.dtype,
            "crs": fm_crs,
            "transform": fm_transform,
        }
        print(f"Writing to {out_path}...")
        with rasterio.open(out_path, "w", **out_kwargs) as dst:
            rasterio.warp.reproject(
                source=cama_data,
                destination=rasterio.band(dst, 1),
                src_transform=cama_transform,
                src_crs=u_const.DEFAULT_CRS,
                dst_transform=fm_transform,
                dst_crs=fm_crs,
                resampling=rasterio.warp.Resampling.bilinear)


def main():
    parser = argparse.ArgumentParser(description='Reproject CaMa-Flood data to flood map areas.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    parser.add_argument('--protection', type=str, metavar="PROT", default="no",
                        choices=["no", "2y", "flopros"],
                        help='Assumption about flood protection.')
    args = parser.parse_args()

    print(f"CaMa data for {args.source} maps...")
    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / args.source / "meta.hdf5")
    for idx, row in meta.iterrows():
        process_map(args.source, args.protection, row)


if __name__ == "__main__":
    main()
