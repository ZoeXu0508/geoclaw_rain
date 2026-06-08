"""
Rename DFO flood maps according to IBTrACS ID and reduce to permanent water and flooded
"""
import argparse
import pathlib
import subprocess

import numpy as np
import pandas as pd
import rasterio
import rasterio.merge

import gcvalid.util.constants as u_const


DFO_DATA_DIR = u_const.FLOODMAPS_DIR / "dfo"

storms_from_geotiffs = [
    {
        "ibtracs_id": "2016273N13300",
        "files": ["20161005USA4402HaitiGrandAnseSUDALOS.tif", "2016USA4402NC.tif"],
        "colors_inundation": [(255, 32, 32), (192, 0, 0)],
        "colors_permanent": [(48, 117, 255)]
    },
    {
        "ibtracs_id": "2017228N14314",
        "files": ["2017USA4510.tif"],
        "colors_inundation": [(224, 0, 0)],
        "colors_permanent": [(48, 117, 255)]
    },
    {
        "ibtracs_id": "2017242N16333",
        "files": [
            "2017USA4516CentralFlorida.tif",
            "2017USA4516SouthFlorida.tif",
            "2017USA4516Cuba.tif",
            "2017USA4516Haiti.tif"
        ],
        "colors_inundation": [(224, 0, 0)],
        "colors_permanent": [(48, 117, 255), (64, 128, 255)]
    },
    {
        "ibtracs_id": "2017253N14130",
        "files": ["2017Vietnam4518a.tif", "2017Vietnam4518b.tif", "2017Vietnam4518_merged.tif"],
        "colors_inundation": [(224, 0, 0), (255, 32, 32)],
        "colors_permanent": [(64, 128, 255)]
    },
    {
        "ibtracs_id": "2017260N12310",
        "files": ["2017USA4523PuertoRico.tif"],
        "colors_inundation": [(224, 0, 0)],
        "colors_permanent": [(48, 117, 255)]
    },
    {
        "ibtracs_id": "2017277N11279",
        "files": ["2017USA4524EAST.tif", "2017USA4524West.tif"],
        "colors_inundation": [(224, 0, 0)],
        "colors_permanent": [(64, 128, 255)]
    },
    {
        "ibtracs_id": "2017304N11127",
        "files": ["2017Vietnam4533.tif"],
        "colors_inundation": [(224, 0, 0), (250, 80, 80)],
        "colors_permanent": [(64, 128, 255)]
    },
    {
        "ibtracs_id": "2018242N13343",
        "files": ["2018USA4676.tif"],
        "colors_inundation": [(255, 32, 32)],
        "colors_permanent": [(0, 85, 255)]
    },
    {
        "ibtracs_id": "2018280N18273",
        "files": ["2018USA4687.tif"],
        "colors_inundation": [(255, 32, 32)],
        "colors_permanent": [(0, 85, 255)]
    },
    {
        "ibtracs_id": "2018292N14261",
        "files": ["2018Mexico4695Willa.tif"],
        "colors_inundation": [(255, 32, 32)],
        "colors_permanent": [(48, 117, 255)]
    },
    {
        "ibtracs_id": "2019063S18038",
        "files": ["2019Malawi4725CombinedLarge.tif"],
        "colors_inundation": [(255, 32, 32), (224, 0, 0)],
        "colors_permanent": [(48, 117, 255)]
    },
    {
        "ibtracs_id": "2019192N29274",
        "files": ["4771.tif"],
        "colors_inundation": [(255, 32, 32)],
        "colors_permanent": [(64, 128, 255)]
    },
    {
        "ibtracs_id": "2019261N28264",
        "files": ["2019USA4797.tif"],
        "colors_inundation": [(255, 32, 32), (224, 0, 0)],
        "colors_permanent": [(64, 128, 255)]
    },
    {
        "ibtracs_id": "2019264N19071",
        "files": ["2019Oman4802.tif"],
        "colors_inundation": [(246, 71, 77), (236, 67, 61), (217, 52, 77)],  # transparent overlay
        "colors_permanent": [(64, 128, 255)]
    },
    {
        "ibtracs_id": "2019329N09160",
        "files": ["2019Philippines4827.tif"],
        "colors_inundation": [(255, 32, 32)],
        "colors_permanent": [(48, 117, 255)]
    },
]

storms_from_images = [
    {
        'ibtracs_id': '2001157N28265',
        'files': ['2001-051Radarsat.tif'],
        'colors_inundation': [(0, 0, 255), (111, 111, 255)],
        'colors_permanent': [(0, 0, 128)],
    },
    {
        'ibtracs_id': '2003196N05150',
        'files': ['2003170Luzon.tif', '2003177Guangxi.tif'],
        'colors_inundation': [(255, 0, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2003249N14329',
        'files': ['2003239Isabel.tif'],
        'colors_inundation': [(255, 0, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2003262N17254',
        'files': ['2003243Mexico.tif'],
        'colors_inundation': [(255, 0, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2004056S18125',
        'files': ['2004033Pilbara.tif'],
        'colors_inundation': [(255, 0, 0), (250, 130, 130), (250, 178, 178)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2004061S12072',
        'files': ['2004045NMad.tif'],
        'colors_inundation': [(246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2004081S15148',
        'files': ['2004047NoQld.tif'],
        'colors_inundation': [(246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2004072S11146',
        'files': ['2004050DeGrey.tif'],
        'colors_inundation': [(246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2004174N14146',
        'files': ['2004105Taiwan.tif'],
        'colors_inundation': [(243, 3, 0), (250, 130, 130), (250, 145, 145)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2004231N09147',
        'files': ['2004143Luzon.tif'],
        'colors_inundation': [(246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2004258N16300',
        'files': ['2004155Forida.tif', '2004155Haiti.tif', '2004155DomRep.tif'],
        'colors_inundation': [(251, 161, 161), (250, 130, 130), (248, 99, 95), (246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2004319N10134',
        'files': ['2004176Phil.tif'],
        'colors_inundation': [(249, 100, 96), (246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2005017S09061',
        'files': ['2005014SMadagascar.tif'],
        'colors_inundation': [(246, 4, 0), (248, 83, 81), (251, 146, 143)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2005236N23285',
        'files': ['2005114Biloxi.tif', '2005114KatrinaSFla.tif', '2005114NewOrleans.tif',
                  '2005114Mobile.tif', '2005114MissDelta.tif'],
        'colors_inundation': [
            (246, 4, 0), (249, 100, 96), (250, 130, 129), (251, 145, 147), (250, 162, 160),
            (253, 177, 177), (253, 209, 208)
        ],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2005257N15120',
        'files': ['2005127NgheAn.tif'],
        'colors_inundation': [(246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2005261N21290',
        'files': ['2005130GulfCoastRita.tif'],
        'colors_inundation': [(246, 4, 0), (249, 100, 96)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2005275N19274',
        'files': ['2005139SMexico.tif', '2005139Retalhuleu.tif', '2005139Escuintla.tif',
                  '2005139CenAmer.tif'],
        'colors_inundation': [(245, 6, 1), (248, 83, 81), (250, 130, 129), (250, 161, 157)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2005289N18282',
        'files': ['2005148WilmaSFla.tif'],
        'colors_inundation': [(246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2005300N10279',
        'files': ['2005154HondNic.tif'],
        'colors_inundation': [
            (246, 4, 0), (248, 36, 32), (249, 100, 96), (251, 146, 143), (252, 178, 177)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2006237N13298',
        'files': ['2006183NeCapeFear.tif'],
        'colors_inundation': [(246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2006240N12265',
        'files': ['2006184SBaja.tif'],
        'colors_inundation': [(246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2007244N12303',
        'files': ['2007177MexPanuco.tif', '2007177HondMotagua.tif', '2007177NicMiskito.tif'],
        'colors_inundation': [(246, 4, 0), (248, 100, 98), (251, 145, 147)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2007297N18300',
        'files': ['2007211DomRep.tif'],
        'colors_inundation': [
            (135, 3, 0), (185, 3, 0), (246, 4, 0), (248, 83, 81), (251, 146, 143)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2007345N18298',
        'files': ['2007232Yaque.tif'],
        'colors_inundation': [(246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2008117N11090',
        'files': ['2008052Burma.tif'],
        'colors_inundation': [(246, 4, 0)],
        'colors_permanent': [(177, 228, 255)],
    },
    {
        'ibtracs_id': '2011233N15301',
        'files': ['SNewJersey.tif'],
        'colors_inundation': [(255, 80, 80)],
        'colors_permanent': [(80, 139, 255)],
    },
    {
        'ibtracs_id': '2012010S24049',
        'files': ['2012Mozambique.tif'],
        'colors_inundation': [(255, 28, 15)],
        'colors_permanent': [(81, 147, 255)],
    },
    {
        'ibtracs_id': '2019116N02090',
        'files': ['2019India4746.tif'],
        'colors_inundation': [(255, 33, 33)],
        'colors_permanent': [(48, 117, 255)],
    },
    {
        'ibtracs_id': '2019112S10053',
        'files': ['2019Mozambique4745.tif'],
        'colors_inundation': [(255, 33, 33)],
        'colors_permanent': [(48, 117, 255)],
    },
]


def jpg_to_geotiff(tif_path):
    jpg_path = tif_path.parent.parent / "images_clean" / f"{tif_path.stem}.jpg"
    if not jpg_path.exists():
        raise FileNotFoundError(f"Crop image manually and store in this location: {jpg_path}")

    points_path = tif_path.parent.parent / "images_gcps" / f"{jpg_path.name}.points"
    if not points_path.exists():
        raise FileNotFoundError(
            f"Use QGis georeferencer and store ground control points here: {points_path}"
        )

    tmp_path = tif_path.parent.parent / "images_tmp" / tif_path.name
    tmp_path.parent.mkdir(exist_ok=True)

    tmp_path.unlink(missing_ok=True)
    tif_path.unlink(missing_ok=True)

    df = pd.read_csv(points_path)
    subprocess.run(
        ["gdal_translate", "-of", "GTiff"]
        + sum([
            ["-gcp", f"{px:.4f}", f"{-py:.4f}", f"{mx:.4f}", f"{my:.4f}"]
            for _, (px, py, mx, my) in df[['pixelX', 'pixelY', 'mapX', 'mapY']].iterrows()
        ], [])
        + [str(jpg_path), str(tmp_path)]
    )
    subprocess.run([
        "gdalwarp", "-r", "near", "-tps", "-co", "COMPRESS=DEFLATE", "-t_srs", "EPSG:4326",
        str(tmp_path), str(tif_path),
    ])


def unify_dfo_maps(mode, ibtracs_ids=None):
    ibtracs_ids = [] if ibtracs_ids is None else ibtracs_ids

    print("Selected mode:", mode)
    storms = storms_from_geotiffs if mode == "from_geotiffs" else storms_from_images

    all_inundation_colors = list(set(sum([st['colors_inundation'] for st in storms], [])))
    all_permanent_colors = list(set(sum([st['colors_permanent'] for st in storms], [])))

    DFO_DATA_DIR.joinpath("geotiff_by_sid").mkdir(exist_ok=True)
    DFO_DATA_DIR.joinpath("images_by_sid").mkdir(exist_ok=True)
    DFO_DATA_DIR.joinpath("clean_by_sid").mkdir(exist_ok=True)

    for storm in storms:
        if len(ibtracs_ids) > 0 and storm['ibtracs_id'] not in ibtracs_ids:
            continue
        if storm['ibtracs_id'] == "2017253N14130":
            infile1, infile2, outfile = storm['files']
            rasterio.merge.merge(
                [DFO_DATA_DIR / "geotiff" / infile1,
                 DFO_DATA_DIR / "geotiff" / infile2],
                dst_path=DFO_DATA_DIR / "geotiff" / outfile)
            storm['files'] = [outfile]
        for i_file, filename in enumerate(storm['files']):
            outpath = DFO_DATA_DIR / "clean_by_sid" / f"{storm['ibtracs_id']}-{i_file}.tif"
            inpath = DFO_DATA_DIR / "geotiff" / filename

            if mode == "from_images":
                img_ext = ".tif"
                if not (DFO_DATA_DIR / "images" / filename).exists():
                    img_ext = ".jpg"
                linkpath = (
                    DFO_DATA_DIR / "images_by_sid" / f"{storm['ibtracs_id']}-{i_file}{img_ext}"
                )
                if not linkpath.exists():
                    # use relative paths in case the base directory changes in the future
                    imgpath = pathlib.Path("..") / "images" / filename.replace(".tif", img_ext)
                    linkpath.symlink_to(imgpath)

                if not inpath.exists():
                    jpg_to_geotiff(inpath)

            linkpath = DFO_DATA_DIR / "geotiff_by_sid" / f"{storm['ibtracs_id']}-{i_file}.tif"
            if not linkpath.exists():
                # use relative paths in case the base directory changes in the future
                linkpath.symlink_to(pathlib.Path("..") / "geotiff" / filename)

            with rasterio.open(inpath, "r") as src:
                data = src.read()[:3,:,:]

            # set inundated colors to 1, permanent to 2, everything else to 0
            inundation_colors = storm['colors_inundation']
            permanent_colors = storm['colors_permanent']
            inundation_mask = np.any([np.abs(data - np.array(col)[:,None,None]).sum(axis=0) < 50
                                      for col in inundation_colors], axis=0)
            permanent_mask = np.any([np.abs(data - np.array(col)[:,None,None]).sum(axis=0) < 50
                                     for col in permanent_colors], axis=0)
            data = np.zeros(data.shape[1:], dtype=np.uint8)
            data[inundation_mask] = 1
            data[permanent_mask] = 2

            kwargs = {
                "driver": "GTiff",
                "compress": "deflate",
                "height": data.shape[0],
                "width": data.shape[1],
                "count": 1,
                "dtype": np.uint8,
                "nodata": 255,
                "crs": src.crs,
                "transform": src.transform,
            }
            with rasterio.open(outpath, "w", **kwargs) as dst:
                print(f"Writing to {outpath}...")
                dst.write(data, 1)


def main():
    parser = argparse.ArgumentParser(description='Preprocess (unify) DFO flood maps.')
    parser.add_argument(
        '--mode', type=str, metavar="MODE", choices=['from_images', "from_geotiffs"],
        help='Whether to process the image (*.jpg) or GeoTIFF (*.tif) files.',
    )
    parser.add_argument(
        "--storms", type=str, nargs="+", metavar="IBTRACS_ID",
        help="IBTrACS IDs of storms to process.",
    )
    args = parser.parse_args()

    if args.mode is None:
        unify_dfo_maps("from_images", ibtracs_ids=args.storms)
        unify_dfo_maps("from_geotiffs", ibtracs_ids=args.storms)
    else:
        unify_dfo_maps(args.mode, ibtracs_ids=args.storms)


if __name__ == "__main__":
    main()
