
import pathlib

import numpy as np

BASE_ARGS = lambda tides, suffix: [
    "2019063S18038",
    "--zos", "aviso",
    "--tides", tides,
    "--bounds", "B31.02839,-20.63320,39.07650,-15.38107",
    "--gauges", "P-20.40417,34.70417", "P-20.20417,34.76250", "P-19.99583,34.77917",
    "P-19.87917,34.78750", "P-19.87083,34.87917", "P-19.84583,34.79583", "P-19.73750,35.08750",
    "P-19.59583,35.27083", "P-19.44583,35.42917", "P-19.29583,35.56250", "P-19.12917,35.71250",
    "P-19.00417,35.87917", "P-18.93750,36.07917", "P-18.90417,36.25417", "P-18.87083,36.32917",
    "P-18.73750,36.40417", "P-18.54583,36.52917", "P-18.38750,36.69583", "P-18.20417,36.84583",
    "P-18.03750,36.98750", "P-18.02083,37.01250", "P-17.87083,37.11250", "P-17.72083,37.27917",
    "P-17.67083,37.37083", "P-17.62083,37.45417", "P-17.50417,37.68750", "P-17.41250,37.86250",
    "P-17.33750,38.07083", "P-17.23750,38.27917", "P-17.16250,38.47917", "P-17.10417,38.69583",
    "P-17.05417,38.93750",
    "--resolution", "9",
    "--suffix", suffix,
]

MOD_INTEN_PERC = [0.0, -8.5, -12.0, -6.0]
MOD_INTEN_RATIO = [1 + p / 100 for p in MOD_INTEN_PERC]

MOD_ZOS_PRIO1 = [0.0] + list(np.around(np.arange(-0.07, -0.19, -0.02), decimals=3))
MOD_ZOS_PRIO2 = [slr for slr in np.around(np.arange(-0.065, -0.175, -0.005), decimals=3)
                 if slr not in MOD_ZOS_PRIO1]
MOD_ZOS = {1: MOD_ZOS_PRIO1, 2: MOD_ZOS_PRIO2}

TIDES = ["no", "mean", "max", "min"]

JOBS_DIR = pathlib.Path("jobs")

def main(prio):
    path = JOBS_DIR / f"idai_cf-prio{prio}.txt"
    out_str = []
    for inten in MOD_INTEN_RATIO:
        for zos in MOD_ZOS[prio]:
            for tides in TIDES:
                if inten == 1:
                    # there are already results without intensity change
                    continue
                suffix = "_cf"
                extra_args = []
                if inten != 1.0:
                    suffix += f"wind{inten * 1000:03.0f}"
                    extra_args += ["--mod_intensity", f"{inten:.3f}"]
                if zos != 0.0:
                    suffix += f"zos{-zos * 1000:03.0f}"
                    extra_args += ["--mod_zos", f"{zos:.3f}"]
                args = BASE_ARGS(tides, suffix) + extra_args
                out_str.append(" ".join(args))
    out_str = "\n".join(out_str)
    path.write_text(out_str)


if __name__ == "__main__":
    for prio in [1, 2]:
        main(prio)
