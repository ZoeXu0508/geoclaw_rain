
import pathlib

import gcvalid.compare.maps as cmp_maps
from gcvalid.sourcedata import data_extents_raster
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def main():
    ref_source = "rapid"
    ibtracs_id = "2017228N14314"
    map_id = f"{ibtracs_id}-0"
    zos = "aviso-fes_max"
    model_thresh = 0.1
    cama_prot = "flopros"
    models = ["rapid", "dfo", "geoclaw", "climada", "aq_codec", "aq_geoclaw"]

    transform, data = data_extents_raster(ref_source, map_id, zos, models, cama_prot, model_thresh)
    for model, d in data.items():
        path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-{model}.tif"
        cmp_maps.write_compare_tif(path, d, transform)


if __name__ == "__main__":
    main()
