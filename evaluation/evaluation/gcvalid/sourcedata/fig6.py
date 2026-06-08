
import pathlib
import pickle

import numpy as np

from gcvalid.sourcedata import data_gauges, data_hwms
import gcvalid.util.constants as u_const


FIGURE_NO = pathlib.Path(__file__).stem[3:]


def print_stats_gauges(df):
    for sim in ["codec", "geoclaw"]:
        _df = df[(df["model"] == sim) & (df["reference"] == "gesla3")]
        print(f"=> {sim}: mean and median signed and abs dev of max water levels")
        for suffix in ["_signed", ""]:
            vals = _df[f"dmax{suffix}"]
            print(f"{vals.mean():.2f}, {vals.median():.2f}")


def data_hwms_storm(source, fes, cama_prot, ibtracs_id):
    df = data_hwms(source, fes, cama_prot)
    df = df[df["record_id"].str.startswith(ibtracs_id)].reset_index(drop=True)
    df = df[df["model"] == "geoclaw"].reset_index(drop=True)
    print(f"N={df.shape[0]} HWM locations")
    return df[["lat", "lon", "hwm_above_gnd_m"]]


def data_gauges_storm(source, fes, ibtracs_id):
    df = data_gauges(source, fes, verbose=False)
    df = df[df["record_id"].str.startswith(ibtracs_id)].reset_index(drop=True)
    print_stats_gauges(df)
    df = df[df["model"] == "geoclaw"].reset_index(drop=True)
    gauge_locations = df[["lon", "lat"]].values
    print(f"N={(df['reference'] == 'gesla3').sum()} GESLA3 locations")
    print(f"N={(df['reference'] == 'codec').sum()} CoDEC locations")
    return df[["stname", "reference", "lon", "lat", "max_obs"]]


def stdata_to_df(stdata, period):
    print(stdata["filename"])
    df = stdata['referenced']
    df.name = stdata["gsrc"]
    df = df.to_frame()
    for sim_data in stdata["simulated"]:
        if sim_data["model"] != "codec" and not sim_data["model"].endswith("_max"):
            continue
        col = "codec" if sim_data["model"] == "codec" else "geoclaw"
        df[col] = sim_data["referenced"]
    df = df[(df.index >= period[0]) & (df.index <= period[1])]
    return df.reset_index()


def data_gauges_single(ibtracs_id, periods):
    path = u_const.COMPARE_DIR / "dfo" / "gauges" / f"{ibtracs_id}-0.pickle"
    with path.open("rb") as fp:
        gaugedata = pickle.load(fp)
    dfs = []
    for stname in ["packery_channel", "high_island", "freshwater_canal_locks"]:
        stdata = [d for d in gaugedata if d["filename"].startswith(stname)][0]
        dfs.append(stdata_to_df(stdata, periods[stdata["filename"]]))
    return dfs


def main():
    source = "all"
    fes = "max"
    cama_prot = "flopros"
    ibtracs_id = "2017228N14314"
    periods = {
        "packery_channel-8775792-usa-noaa": (
            np.datetime64("2017-08-25T04:00:00"),
            np.datetime64("2017-08-27T20:00:00"),
        ),
        "high_island-8770808-usa-noaa": (
            np.datetime64("2017-08-29T00:00:00"),
            np.datetime64("2017-08-31T05:00:00"),
        ),
        "freshwater_canal_locks-8766072-usa-noaa": (
            np.datetime64("2017-08-29T00:00:00"),
            np.datetime64("2017-08-31T05:00:00"),
        ),
        # for testing purposes, some more stations:
        "port_aransas-8775237-usa-noaa": (
            np.datetime64("2017-08-25T04:00:00"),
            np.datetime64("2017-08-27T20:00:00"),
        ),
        "port_arthur-8770475-usa-noaa": (
            np.datetime64("2017-08-29T00:00:00"),
            np.datetime64("2017-08-31T05:00:00"),
        ),
    }

    df = data_hwms_storm(source, fes, cama_prot, ibtracs_id)
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}a.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)

    df = data_gauges_storm(source, fes, ibtracs_id)
    path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}b.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)

    dfs = data_gauges_single(ibtracs_id, periods)
    for panel, df in zip(["c", "d", "e"], dfs):
        path = u_const.SOURCEDATA_DIR / f"fig{FIGURE_NO}{panel}.csv"
        print(f"Writing to {path} ...")
        df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
