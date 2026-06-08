
import pathlib
import sys

import numpy as np
import pandas as pd


LOG_PATH = pathlib.Path("log")


def extract_time_from_logline(line):
    trunc = line[:23].replace(",", ".")
    return np.datetime64(trunc)


def runtimes_from_logfile(logfile):
    lines = [l for l in logfile.read_text().split("\n") if l[:2] == "20"]
    timings = [
        ("total", extract_time_from_logline(lines[0]), extract_time_from_logline(lines[-1]))
    ]
    name = start = None
    for l in lines:
        if "Running GeoClaw in " in l:
            start = extract_time_from_logline(l)
            name = l.split("/")[-1].split(" ")[0]
        elif None not in [name, start] and "Reading GeoClaw output" in l:
            timings.append(
                (name, start, extract_time_from_logline(l))
            )
            name = start = None
    timings = pd.DataFrame(timings, columns=["gc_run_id", "start", "end"])
    timings["runtime_sec"] = (timings["end"] - timings["start"]) / np.timedelta64(1, 's')
    return timings


def human_readable_timedelta(timedelta_sec):
    if timedelta_sec < 60:
        time_str = f"{timedelta_sec:.0f}sec"
    elif timedelta_sec < 60 * 60:
        time_str = f"{timedelta_sec / 60:.0f}min"
    else:
        time_str = f"{timedelta_sec / 3600:.0f}h{(timedelta_sec / 60) % 60:.0f}min"
    return time_str


def main():
    data = []
    for logfile in LOG_PATH.glob("*.out"):
        args = logfile.parent / logfile.name.replace(".out", ".args")
        args = args.read_text().strip()
        df = runtimes_from_logfile(logfile)
        df["job_id"] = logfile.stem.split("-")[1]
        df["ibtracs_id"] = args[:13]
        df["source"] = args.split("--suffix")[1][2:].strip()
        data.append(df)
    df = pd.concat(data)
    df["runtime_str"] = df["runtime_sec"].apply(human_readable_timedelta)

    df_total = df[df["gc_run_id"] == "total"].sort_values(by="runtime_sec").reset_index(drop=True)
    df_runs = df[df["gc_run_id"] != "total"].sort_values(by="runtime_sec").reset_index(drop=True)

    for df in [df_total, df_runs]:
        qs = [0.1, 0.33, 0.5, 0.6, 0.66, 0.9]
        stats = (
            [df["runtime_sec"].mean()]
            + df["runtime_sec"].quantile(q=qs).tolist()
        )
        print(" ".join(["    mean"] + [f"{100 * q:6.0f}th" for q in qs]))
        print(" ".join(f"{s:8.0f}" for s in stats))
        print(" ".join(f"{human_readable_timedelta(s):>8s}" for s in stats))
        print(df[["job_id", "ibtracs_id", "source", "gc_run_id", "runtime_str"]])
        print()


if __name__ == "__main__":
    main()
