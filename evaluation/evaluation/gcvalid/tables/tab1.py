
import pathlib

import numpy as np
import pandas as pd

from gcvalid.figures import MODEL_NAMES
import gcvalid.util.constants as u_const


TABLE_NO = pathlib.Path(__file__).stem[3:]


def tex_extent_stats(df, l_models):
    indicator_names = {
        "mcc": "MCC",
        "f1": "F1",
        "f2": "F2",
        "tnr": r"TNR (\%)",
        "bias": "Bias",
    }
    inds = list(indicator_names.keys())
    table_str = (
        r"Model & "
    ) + " & ".join(indicator_names.values()) + (
        r" \\\midrule"
        "\n"
    )
    for model in l_models:
        name = MODEL_NAMES[model]
        df_m = df[df["model"] == model].set_index("indicator")
        table_str += r"\multirow{2}{*}{" f"{name}" r"} & "
        df_m.loc["tnr", :] *= 100
        totals, mids, lows, highs = [
            df_m.loc[inds, c] for c in ["total", "median", "17", "83"]
        ]

        # replace very low log-values by "-inf"
        if lows["bias"] < -10:
            lows["bias"] = -np.inf

        table_str += " & ".join(
            f"{total:+.2f}" + r" $\filledstar$ " + f"{mid:+.2f}"
            if ind in ["mcc", "bias"] else
            f"{total:.2f}" + r" $\filledstar$ " + f"{mid:.2f}"
            if ind != "tnr" else
            f"{total:.0f}\\%" + r" $\filledstar$ " + f"{mid:.0f}\\%"
            for ind, mid, total in zip(inds, mids, totals)
        )
        table_str += r" \\" + "\n& "

        table_str += " & ".join(
            f"({low:+.2f} -- {high:+.2f})"
            if ind in ["mcc", "bias"] else
            f"({low:.2f} -- {high:.2f})"
            if ind != "tnr" else
            f"({low:.0f}\\% -- {high:.0f}\\%)"
            for ind, low, high in zip(inds, lows, highs)
        )
        mrule = "" if model == l_models[-1] else r"\midrule"
        table_str += r" \\" + f"{mrule}\n"
    return table_str


def main():
    models = ['geoclaw', 'aq_geoclaw', 'aq_codec', 'climada']

    df = pd.read_csv(u_const.SOURCEDATA_DIR / "fig4.csv")
    s_tex = tex_extent_stats(df, models)

    path = u_const.TABLES_DIR / f"tab{TABLE_NO}.tex"
    print(f"Writing to {path} ...")
    path.write_text(s_tex)


if __name__ == "__main__":
    main()
