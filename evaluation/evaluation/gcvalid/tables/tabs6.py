
import pathlib

import numpy as np
import pandas as pd

from gcvalid.figures import MODEL_NAMES
import gcvalid.util.constants as u_const


TABLE_NO = pathlib.Path(__file__).stem[3:]


def tex_mixed_stats(df, l_models):
    indicator_names = {
        "mcc": "MCC",
        "f1": "F1",
        "f2": "F2",
        "tnr": r"TNR",
        "bias": "Bias",
        "model_flooded": "Hit rate",
        "dinund": "Error (m)",
        "dinund_signed": "Bias (m)",
    }
    inds = list(indicator_names.keys())

    table_str = "& Indicator & " + (
        " & ".join([MODEL_NAMES[m] for m in l_models])
    ) + (r" \\\midrule" + "\n")

    for i_ind, (ind, ind_name) in enumerate(indicator_names.items()):
        if i_ind == 0:
            table_str += r"\multirow{10}{*}{\rotatebox{90}{Flood extents}} "
        elif i_ind == 5:
            table_str += r"\multirow{6}{*}{\rotatebox{90}{HWMs}} "
        table_str += r"& \multirow{2}{*}{" f"{ind_name}" r"} & "

        df_i = df[df["indicator"] == ind].set_index("model")
        totals, meds, lows, highs = [
            df_i.loc[l_models, c] for c in ["total", "median", "17", "83"]
        ]

        # replace very low log-values by "-inf"
        if ind == "bias":
            lows[lows < -10] = -np.inf

        table_str += " & ".join(
            f"{total:+.2f}" + r" $\filledstar$ " + f"{med:+.2f}"
            if ind in ["mcc", "bias"] or ind.endswith("_signed") else
            f"{total:.0f}\\%" + r" $\filledstar$ " + f"{med:.0f}\\%"
            if ind == "tnr" else
            (r"\multirow{2}{*}{" f"{total:.1f}" r"\%}")
            if ind == "model_flooded" else
            f"{total:.2f}" + r" $\filledstar$ " + f"{med:.2f}"
            for med, total in zip(meds, totals)
        )
        table_str += r" \\" + "\n&& "

        table_str += " & ".join(
            f"({low:+.2f} -- {high:+.2f})"
            if ind in ["mcc", "bias"] or ind.endswith("_signed") else
            f"({low:.0f}\\% -- {high:.0f}\\%)"
            if ind == "tnr" else
            ""
            if ind == "model_flooded" else
            f"({low:.2f} -- {high:.2f})"
            for low, high in zip(lows, highs)
        )

        mrule = (
            ""
            if i_ind == len(indicator_names) - 1 else
            r"\cmidrule(lr){2-5}"
            if i_ind != 4 else
            r"\midrule"
        )
        table_str += r" \\" + f"{mrule}\n"
    return table_str


def main():
    models = ['geoclaw', 'geoclaw+cama', 'cama']

    df_extents = pd.read_csv(u_const.SOURCEDATA_DIR / "figs6-extents.csv")
    df_hwms = (
        pd.read_csv(u_const.SOURCEDATA_DIR / "figs6-hwms.csv")
        .rename(columns={"mean": "total"})
    )
    df = pd.concat([df_extents, df_hwms]).drop(columns=["mean"])
    df.loc[
        df["indicator"].isin(["model_flooded", "tnr"]),
        ["total", "median", "17", "83"]
    ] *= 100
    s_tex = tex_mixed_stats(df, models)

    path = u_const.TABLES_DIR / f"tab{TABLE_NO}.tex"
    print(f"Writing to {path} ...")
    path.write_text(s_tex)


if __name__ == "__main__":
    main()
