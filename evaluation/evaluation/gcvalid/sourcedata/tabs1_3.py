
import pathlib

from climada.hazard import TCTracks
import pandas as pd

from gcvalid.sourcedata import data_extents
from gcvalid.compare.hwm import read_hwms
import gcvalid.util as util
import gcvalid.util.constants as u_const
import gcvalid.util.gauge as u_gauge


TABLE_NO = pathlib.Path(__file__).stem[3:]


def data_events():
    df = data_extents("all", "max", "flopros", 0.1)
    df["region"] = "AP"
    df.loc[df["lon_mean"] > 0, "region"] = "PI"
    df.loc[df["lat_mean"] < 0, "region"] = "SH"
    df["area_flooded"] = df["tp"] + df["fn"]
    df["sshs"] = util.saffir_simpson_category(df["maxwind"].values)
    df = (
        df[df["model"] == "geoclaw"]
        .drop(columns=[
            "model", "record_id", "lon_mean", "lat_mean", "tp", "fp", "tn", "fn", "maxwind"
        ])
    )

    df_meta = []
    for src in ["dfo", "gfd", "rapid"]:
        path = u_const.FLOODMAPS_DIR / src / "meta.hdf5"
        _df = pd.read_hdf(path).sort_values(by="map_id")
        _df["source"] = src
        df_meta.append(_df)
    df_meta = (
        pd.concat(df_meta)
        .groupby(by=["source", "ibtracs_id"])
        ["date"]
        .apply(lambda v: ",".join(v))
        .reset_index()
    )
    df = df.merge(df_meta, on=["source", "ibtracs_id"], how="left")

    df_names = (
        pd.DataFrame([
            (tr.attrs["sid"], tr.attrs["ibtracs_name"])
            for src in ["dfo", "gfd", "rapid"]
            for tr in TCTracks.from_netcdf(u_const.TRACKS_DIR / src).data
        ], columns=["ibtracs_id", "name"])
        .groupby("ibtracs_id")
        .first()
        .reset_index()
    )
    df = df.merge(df_names, on="ibtracs_id", how="left")

    for gsrc in ["gesla3", "codec"]:
        df_gauges = []
        for src in ['gfd', 'dfo', 'rapid']:
            _df = u_gauge.load_gauge_locations(src, gsrc)
            _df["ibtracs_id"] = _df["map_id"].str.split("-", expand=True)[0]
            df_gauges.append(_df)
        df_gauges = (
            pd.concat(df_gauges)
            .rename(columns={"name": f"n_{gsrc}"})
            .groupby(by=["ibtracs_id"])
            [f"n_{gsrc}"].size()
            .reset_index()
        )
        df = df.merge(df_gauges, on=["ibtracs_id"], how="left")
        df[f"n_{gsrc}"] = df[f"n_{gsrc}"].fillna(0)

    df_hwms = (
        pd.concat([
            read_hwms(src, add_flooded_status=False, as_dataframe=True)
            for src in ['gfd', 'dfo', 'rapid']
        ])
        .rename(columns={
            "fm_source": "source",
            "latitude": "lat",
            "longitude": "lon",
            "hwm_id": "n_hwm",
        })
        .groupby(by=["ibtracs_id", "hwm_environment"])
        ["n_hwm"].size()
        .reset_index()
        .pivot(
            columns=["hwm_environment"],
            index=["ibtracs_id"],
            values="n_hwm",
        )
        .reset_index()
        .rename(columns={
            "Coastal": "n_hwm_coastal",
            "Riverine": "n_hwm_riverine",
        })
    )
    df = df.merge(df_hwms, on=["ibtracs_id"], how="left")
    for c in ["coastal", "riverine"]:
        df[f"n_hwm_{c}"] = df[f"n_hwm_{c}"].fillna(0)

    return df


def main():
    df = data_events()
    path = u_const.SOURCEDATA_DIR / f"tab{TABLE_NO}.csv"
    print(f"Writing to {path} ...")
    df.to_csv(path, index=None)


if __name__ == "__main__":
    main()
