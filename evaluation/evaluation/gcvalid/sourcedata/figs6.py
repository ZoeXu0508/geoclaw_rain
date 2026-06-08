
import pathlib

import gcvalid.compare.maps as cmp_maps
from gcvalid.sourcedata import (
    data_extents,
    data_extents_raster,
    stats_extents,
    stats_hwms,
    print_stats_extents,
)
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def main():
    model_thresh = 0.1
    cama_prot = "flopros"
    fes_setting = "max"

    transform, data = data_extents_raster(
        "rapid", "2017228N14314-0", "aviso-fes_max", ["cama"], cama_prot, model_thresh,
    )
    for model, d in data.items():
        path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-{model}.tif"
        cmp_maps.write_compare_tif(path, d, transform)

    models = ["geoclaw", "geoclaw+cama", "cama"]
    df = data_extents("all", fes_setting, cama_prot, model_thresh)
    df = df[df["model"].isin(models)].copy()
    print_stats_extents(df)
    df = stats_extents(df)
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-extents.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)

    df = stats_hwms("all", fes_setting, cama_prot, models, riverine=True)
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}-hwms.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
