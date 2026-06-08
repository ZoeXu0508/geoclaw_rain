
import pathlib

import numpy as np
import pandas as pd

import gcvalid.util.constants as u_const


TABLE_NO = pathlib.Path(__file__).stem[3:]


def tex_cmp_data(df):
    cols = [
        "IBTrACS ID",
        "Name (Year)",
        "SSHS",
        "Date (MM/DD)",
        "GESLA3",
        "CoDEC",
        "Coastal",
        "Riverine",
        "DFO",
        "GFD",
        "RAPID",
    ]
    n_cols = len(cols)

    table_str = (
        "&" * (n_cols - 5)
        + r" \multicolumn{2}{c}{HWMs} &"
        + r" \multicolumn{3}{c}{Flooded area (km$^2$)} \\"
        + r"\cmidrule(lr){" + f"{n_cols - 4}-{n_cols - 3}" + "}"
        + r"\cmidrule(lr){" + f"{n_cols - 2}-{n_cols}" + "}"
        + "\n"
    ) + " & ".join(cols) + r" \\\midrule" + "\n"
    ibtracs_ids = np.unique(df["ibtracs_id"].values)
    for sid in ibtracs_ids:
        df_s = df[df["ibtracs_id"] == sid]
        name = df_s["name"].values[0].replace("Eline:Leone", "Leon–Eline")
        dates = np.unique(",".join(df_s["date"].values).split(","))
        dates = [d.split("-") for d in dates]
        date = "--".join(f"{d[1]}/{d[2]}" for d in (
            [dates[0], dates[-1]] if len(dates) > 1 else dates
        ))
        year = dates[0][0]
        sshs = df_s["sshs"].values.max()
        area = {}
        for src in ["dfo", "gfd", "rapid"]:
            s_area = df_s.loc[df_s["source"] == src, "area_flooded"].values
            area[src] = "--" if s_area.size == 0 else f"{s_area[0]:.0f}"
        table_str += " & ".join([
            sid,
            f"{name} ({year})",
            u_const.SAFFIR_SIMPSON_NAMES[sshs + 1],
            date,
            f"{df_s['n_gesla3'].values[0]:.0f}",
            f"{df_s['n_codec'].values[0]:.0f}",
            f"{df_s['n_hwm_coastal'].values[0]:.0f}",
            f"{df_s['n_hwm_riverine'].values[0]:.0f}",
            area["dfo"],
            area["gfd"],
            area["rapid"],
        ]) + r" \\" + "\n"
    return table_str


def main():
    df = pd.read_csv(u_const.SOURCEDATA_DIR / f"tab{TABLE_NO}.csv")
    for i_region, region in enumerate(["AP", "PI", "SH"]):
        s_tex = tex_cmp_data(df[df["region"] == region].copy())
        path = u_const.TABLES_DIR / f"tabs{1 + i_region}.tex"
        print(f"Writing to {path} ...")
        path.write_text(s_tex)


if __name__ == "__main__":
    main()
