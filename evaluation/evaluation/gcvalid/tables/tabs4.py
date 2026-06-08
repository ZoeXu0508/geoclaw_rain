
import pathlib

import numpy as np
import pandas as pd

from gcvalid.figures import MODEL_NAMES
import gcvalid.util.constants as u_const


TABLE_NO = pathlib.Path(__file__).stem[3:]


def tex_hwm_stats(df, l_models):
    indicator_names = {
        "model_flooded": "Hit rate (\%)",
        "dinund": "Error (m)",
        "dinund_signed": "Bias (m)",
    }
    inds = list(indicator_names.keys())
    table_str = (
        r"&&& \multicolumn{2}{c}{Elevation / Inundation height}"
        r" \\\cmidrule(lr){4-5}"
        "\nModel & N & "
    ) + " & ".join(indicator_names.values()) + (
        r" \\\midrule"
        "\n"
    )
    for model in l_models:
        name = MODEL_NAMES[model]
        df_mod = df[df["model"] == model].set_index("indicator")
        df_mod.loc["model_flooded", ["mean", "median", "17", "83"]] *= 100
        table_str += r"\multirow{2}{*}{" f"{name}" r"} & "
        counts, means, meds, lows, highs = [
            df_mod.loc[inds, c] for c in ["N", "mean", "median", "17", "83"]
        ]

        table_str += r"\multirow{2}{*}{" f"{counts.values[-1]:.0f}" r"} & "

        table_str += " & ".join(
            r"\multirow{2}{*}{--}"
            if (
                model == "floodmap" and ind != "model_flooded"
                or model == "dem" and ind == "model_flooded"
            ) else
            f"{mn:+.2f} $\\filledstar$ {med:+.2f}"
            if ind.endswith("_signed") else
            f"{mn:.2f} $\\filledstar$ {med:.2f}"
            if ind != "model_flooded" else
            (r"\multirow{2}{*}{" f"{mn:.1f}" r"\%}")
            for ind, mn, med in zip(inds, means, meds)
        )
        table_str += r" \\" + "\n&& "
        table_str += " & ".join(
            ""
            if (
                model == "floodmap" and ind != "model_flooded"
                or model == "dem" and ind == "model_flooded"
            ) else
            f"({low:+.2f} -- {high:+.2f})"
            if ind.endswith("_signed") else
            f"({low:.2f} -- {high:.2f})"
            if ind != "model_flooded" else
            ""
            for ind, low, high in zip(inds, lows, highs)
        )

        mrule = "" if model == "climada" else r"\midrule"
        table_str += r" \\" + f"{mrule}\n"
    return table_str


def main():
    models = ['floodmap', 'dem', 'geoclaw', 'aq_geoclaw', 'aq_codec', 'climada']

    df = pd.read_csv(u_const.SOURCEDATA_DIR / "fig7.csv")
    s_tex = tex_hwm_stats(df, models)

    path = u_const.TABLES_DIR / f"tab{TABLE_NO}.tex"
    print(f"Writing to {path} ...")
    path.write_text(s_tex)


if __name__ == "__main__":
    main()
