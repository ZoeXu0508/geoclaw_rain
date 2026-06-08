
import pathlib

import numpy as np
import pandas as pd

import gcvalid.util.constants as u_const


TABLE_NO = pathlib.Path(__file__).stem[3:]


def tex_gauge_stats(df):
    df["config"] = df["reference"] + "_" + df["model"]
    config_names = {
        "gesla3_geoclaw": ("GeoClaw", "GESLA3"),
        "gesla3_codec": ("GTSM", "GESLA3"),
        "codec_geoclaw": ("GeoClaw", "GTSM"),
    }
    indicator_names = {
        "dmax": "Absolute (m)",
        "dmax_signed": "Signed (m)",
        "pearson": "Pearson",
        "rmse": "RMSE (m)",
    }
    inds = list(indicator_names.keys())
    table_str = (
        r"\multirow{2}{*}{Model}"
        r" & \multirow{2}{*}{Reference}"
        r" & \multicolumn{2}{c}{Deviation of max. sea levels}"
        r" & \multicolumn{2}{c}{Surge dynamics}"
        r" \\\cmidrule(lr){3-4}\cmidrule(lr){5-6}"
        "\n&& "
    ) + " & ".join(indicator_names.values()) + (
        r" \\\midrule"
        "\n"
    )
    for cfg, (name_m, name_r) in config_names.items():
        df_cfg = df[df["config"] == cfg].set_index("indicator")
        table_str += (
            r"\multirow{2}{*}{"
            f"{name_m}"
            r"} & \multirow{2}{*}{"
            f"{name_r}"
            r"} & "
        )
        means, meds, lows, highs = [
            df_cfg.loc[inds, c] for c in ["mean", "median", "17", "83"]
        ]
        table_str += " & ".join(
            f"{mn:+.2f} $\\filledstar$ {med:+.2f}"
            if ind.endswith("_signed") else
            f"{mn:.2f} $\\filledstar$ {med:.2f}"
            for ind, mn, med in zip(inds, means, meds)
        )
        table_str += r" \\" + "\n&& "
        table_str += " & ".join(
            f"({low:+.2f} -- {high:+.2f})"
            if ind.endswith("_signed") else
            f"({low:.2f} -- {high:.2f})"
            for ind, low, high in zip(inds, lows, highs)
        )
        mrule = "" if cfg == "codec_geoclaw" else r"\midrule"
        table_str += r" \\" + f"{mrule}\n"
    return table_str


def main():
    df = pd.read_csv(u_const.SOURCEDATA_DIR / "fig8.csv")
    s_tex = tex_gauge_stats(df)

    path = u_const.TABLES_DIR / f"tab{TABLE_NO}.tex"
    print(f"Writing to {path} ...")
    path.write_text(s_tex)


if __name__ == "__main__":
    main()
