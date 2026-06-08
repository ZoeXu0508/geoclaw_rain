
import numpy as np
import pandas as pd

from gcvalid.compare.hwm import read_hwms
import gcvalid.compare.maps as cmp_maps
import gcvalid.util.constants as u_const
import gcvalid.util.io as u_io


MODEL_SHORTS = {
    "geoclaw": "gc",
    "geoclaw+cama": "gm",
    "cama": "ca",
    "climada": "cl",
    "aq_codec": "ac",
    "aq_geoclaw": "ag",
}

HWMS_FILTERED_PANEL_CONFIGS = [
    (False, False, False),
    (True, True, False),
    (True, False, False),
    (True, False, True),
]

HWM_COMPARE_MODELS = [
    "floodmap", "geoclaw", "geoclaw+cama", "cama", "climada", "aq_codec", "aq_geoclaw"
]

GAUGE_METRICS = ["dmax_signed", "dmax", "dmaxrel_signed", "dmaxrel", "pearson", "rmse"]


def data_extents(source, fes, cama_prot, thresh):
    model_trans = {
        "without": "geoclaw",
        f"w_isimip3a{cama_prot}": "geoclaw+cama",
        f"bt_aq_geoclaw-fes_{fes}": "aq_geoclaw",
        "bt_aq_codec": "aq_codec",
        "bt_climada": "climada",
        f"o_isimip3a{cama_prot}": "cama",
    }
    if "_" in source:
        fm_comparison = source.split("_")[1]
        model_trans[f"fm_{fm_comparison}"] = f"fm_{fm_comparison}"
    df = (
        pd.concat([
            u_io.read_compare_df(
                src, pluv, f"aviso-fes_{fes}", thresh,
                apply_filters=True, verbose=False,
            )
            for src in (['gfd', 'dfo', 'rapid'] if source == "all" else [source])
            for pluv in model_trans.keys()
        ])
        .rename(columns={
            "pluvial": "model",
            "coastal_both_area": "tp",
            "coastal_gc_area": "fp",
            "coastal_fm_area": "fn",
        })
        .reset_index(drop=True)
    )

    # 2023-12-05: Tried to find examples of flood maps with few rain. Sorted by
    # "rainf_ERA5-combined" and printed top 3:
    #   2007345N18298 (300), 2003249N14329 (500), 2005261N21290 (550)
    # When considering wind, these top-10 storms have more than 50 m/s:
    #   2003249N14329, 2005261N21290, 2005236N23285, 2017260N12310, 2019116N02090, 2018280N18273
    # Apart from RITA (2005261N21290) and KATRINA (2005236N23285), none of the flood maps
    # really looks like surge was the main driver. But the flood map of MARIA (2017260N12310) looks
    # quite interesting with actual tidal surge all around Puerto Rico, a lot of flooding, and also
    # HWM data.

    df["record_id"] = df["source"] + "_" + df["ibtracs_id"]
    df["model"] = df["model"].apply(lambda m: model_trans[m])
    df["tn"] = df["coastal_area"] - df["coastal_flooded_area"]
    df_grpby = df.groupby(["record_id", "model"])
    df = df_grpby.first()
    df["maxwind"] = df_grpby["maxwind"].max()
    for col in ["tp", "fp", "fn", "tn"]:
        df[col] = df_grpby[col].sum()
    for c in ["lon", "lat"]:
        df[f"{c}_min"] = df_grpby[f"{c}_min"].min()
        df[f"{c}_max"] = df_grpby[f"{c}_max"].max()
        df[f"{c}_mean"] = 0.5 * (df[f"{c}_min"] + df[f"{c}_max"])
    df = df.reset_index()
    return df[[
        "source", "model", "record_id", "ibtracs_id", "lon_mean", "lat_mean", "maxwind",
        "tp", "tn", "fp", "fn",
    ]]


def _load_raster(model, cama_prot, model_thresh, *args):
    pluvial = {
        "rapid": "fm_rapid",
        "dfo": "fm_dfo",
        "geoclaw": "without",
        "climada": "bt_climada",
        "aq_codec": "bt_aq_codec",
        "aq_geoclaw": "bt_aq_geoclaw-fes_max",
        "cama": f"o_isimip3a{cama_prot}",
    }[model]
    data = cmp_maps.load_raster_data(*args, pluvial, extra=False)[0]
    data[data >= model_thresh] = 1
    data[data < model_thresh] = 0
    data[~np.isfinite(data)] = 2
    data = data.astype(int)
    return data


def _mask_raster(data, waterbody_data, dem_data, wind_data):
    data = data.copy()
    data[waterbody_data > 5] = 255
    data[(dem_data > 10)] = 10
    return data


def data_extents_raster(ref_source, map_id, zos, models, cama_prot, model_thresh):
    _data, bounds, transform = cmp_maps.load_fm_data(ref_source, map_id, None)
    shape = _data.shape

    mod_args = (ref_source, map_id, zos, bounds, shape, transform)
    _, waterbody_data, dem_data, wind_data = cmp_maps.load_raster_data(*mod_args, "without")

    data = {}
    for model in models:
        d = _load_raster(model, cama_prot, model_thresh, *mod_args)
        d = _mask_raster(d, waterbody_data, dem_data, wind_data)
        data[model] = d
    return transform, data


def data_gauges(source, fes, verbose=True):
    gc_model = f"geoclaw-zos_aviso-fes_{fes}"
    df = pd.concat([
        pd.read_csv(u_const.COMPARE_DIR / src / "gauges" / "stats.csv")
        for src in (['gfd', 'dfo', 'rapid'] if source == "all" else [source])
    ]).rename(columns={"gsrc": "reference"})
    df = df[
        np.isin(df["reference"], ["codec", "gesla3"])
        & ((df["model"] == gc_model) & df['valid'] | (df['model'] == "codec"))
        & (~df['rmse'].isna())
    ]
    df.loc[df["model"] == gc_model, "model"] = "geoclaw"
    df["ibtracs_id"] = df["map_id"].str.split("-", expand=True)[0]
    df["record_id"] = df["ibtracs_id"] + "_" + df["stname"]

    for metric in GAUGE_METRICS:
        if metric not in ["pearson", "dmaxrel", "dmaxrel_signed"]:
            # convert mm to m
            df[metric] /= 1000
        if metric.endswith("_signed"):
            # negative values should mean an underestimation by the simulation in the plots:
            df[metric] *= -1

    # avoid having a single record multiple times (can happen if we have several flood extents for
    # that event and location, e.g. from DFO + GFD)
    return (
        df
        .groupby(["reference", "model", "record_id"])
        .first()
        .reset_index()
    )


def _read_hwms(source, fes, cama_prot):
    gc_model = f"geoclaw-fes_{fes}"
    df = (
        pd.concat([
            read_hwms(src, add_flooded_status=True, as_dataframe=True)
            for src in (['gfd', 'dfo', 'rapid'] if source == "all" else [source])
        ])
        .rename(columns={
            "longitude": "lon",
            "latitude": "lat",
            "hwmQualityName": "hwm_quality",
            "elev_m": "hwm_above_geoid_m",
            "height_above_gnd_m": "hwm_above_gnd_m",
            "bt_climada": "climada",
            "bt_aq_codec": "aq_codec",
            f"bt_aq_{gc_model}": "aq_geoclaw",
            gc_model: "geoclaw",
            f"cama_{cama_prot}": "cama",
        })
    )

    df["geoclaw+cama"] = np.fmax(df["cama"], df["geoclaw"])
    groupby_cols = ["ibtracs_id", "hwm_id"]
    df = (
        df
        .groupby(by=groupby_cols)
        .agg({
            col: "mean" if col in HWM_COMPARE_MODELS else "first"
            for col in df.columns
            if col not in groupby_cols
        })
        .reset_index()
    )
    df["record_id"] = df["ibtracs_id"] + "_" + df["hwm_id"].astype(str)
    return df


def data_hwms(
    source,
    fes,
    cama_prot,
    h_above_gnd=False,
    exclude_zeros=True,
    exclude_by_dtopo=False,
    riverine=False,
):
    df = _read_hwms(source, fes, cama_prot)
    df = df[df["hwm_environment"] == ("Riverine" if riverine else "Coastal")].copy()

    ds = (
        df
        .melt(
            id_vars=["record_id"],
            value_vars=HWM_COMPARE_MODELS,
            var_name="model",
            value_name="model_above_gnd_m",
        )
        .set_index(["record_id", "model"])
        .to_xarray()
    )
    for v in ["hwm_above_gnd_m", "hwm_above_geoid_m", "hwm_quality", "dem", "lon", "lat"]:
        ds[v] = df.set_index("record_id")[v]
    ds["model_above_geoid_m"] = ds["dem"] + ds["model_above_gnd_m"]
    ds["model_flooded"] = (ds["model_above_gnd_m"] > 0)

    hwm_uncertainty = {
        'Excellent: +/- 0.05 ft': 0.015,
        'Good: +/- 0.10 ft': 0.03,
        'Fair: +/- 0.20 ft': 0.06,
        'Poor: +/- 0.40 ft': 0.12,
        'VP: > 0.40 ft': 0.15,
        'Unknown/Historical': 0.15,
    }
    ds["hwm_uncertainty_m"] = ("record_id", [hwm_uncertainty[v] for v in ds["hwm_quality"].values])

    ds["hwm_has_topo"] = np.isfinite(ds["hwm_above_geoid_m"]) & np.isfinite(ds["hwm_above_gnd_m"])
    ds["hwm_topo_m"] = ds["hwm_above_geoid_m"].where(ds["hwm_has_topo"])
    ds["hwm_above_geoid_m"] = (
        (ds["hwm_above_geoid_m"] + ds["hwm_above_gnd_m"]).where(
            ds["hwm_has_topo"],
            other=ds["hwm_above_geoid_m"],
        )
    )
    ds["dtopo_signed"] = ds["dem"] - ds["hwm_topo_m"]
    ds["dtopo"] = np.abs(ds["dtopo_signed"])
    ds["dtopo_gt_inund"] = (ds["hwm_above_geoid_m"] + ds["hwm_uncertainty_m"] <= ds["dem"])

    base = "gnd" if h_above_gnd else "geoid"
    ds["dinund_signed"] = (
        (ds[f"model_above_{base}_m"] - ds[f"hwm_above_{base}_m"]).where(
            (ds["model_above_gnd_m"] > 0)
            if exclude_zeros else
            (ds["model_above_gnd_m"] >= 0) & (ds["dtopo"] < 0.5 * ds[f"hwm_above_gnd_m"])
            if exclude_by_dtopo else
            (ds["model_above_gnd_m"] >= 0)
        )
    )

    ds["dinund"] = np.abs(ds["dinund_signed"])

    return ds[[
        "lon", "lat", "hwm_topo_m", "dem", "dtopo", "dtopo_signed", "dtopo_gt_inund", "dinund",
        "dinund_signed", "model_flooded", "hwm_above_gnd_m", "hwm_above_geoid_m",
        "model_above_gnd_m", "model_above_geoid_m",
    ]].to_dataframe().reset_index()


def print_stats(title, data, total=False):
    qs = [0.01, 0.05, 0.17, 0.5, 0.83, 0.95, 0.99]
    print(f"=> {title}")
    cols = (
        [" " * 6, "    N"]
        + ([" Total"] if total else [])
        + [f"{s:>6s}" for s in ["Mean", "Min", "Max"]]
         + [f"{100 * q:>6.0f}" for q in qs]
    )
    print(" ".join(cols))
    print("-" * (7 * len(cols) - 1))
    for name, d in data.items():
        dq = [d.mean(), d.min(), d.max()] + [d.quantile(q=q) for q in qs]
        if total:
            dq = [d.sum()] + dq
        print(f"{name:<6s} " + f"{d.size:5d} " + " ".join(f"{v:6.2f}" for v in dq))
    print()


def print_stats_extents(df):
    df = df.copy()

    world_regions = [
        ('AP', (-180, 0, 0, 90)),
        ('PI', (0, 0, 180, 90)),
        ('SH', (-180, -90, 180, 0)),
    ]
    df["region"] = ""
    for region_name, region_bounds in world_regions:
        mask = (
            (df["lon_mean"] >= region_bounds[0])
            & (df["lon_mean"] <= region_bounds[2])
            & (df["lat_mean"] >= region_bounds[1])
            & (df["lat_mean"] <= region_bounds[3])
        )
        df.loc[mask, "region"] = region_name

    df[["tp", "fp", "tn", "fn"]] /= 1000
    df["total_area"] = df[["tp", "fp", "tn", "fn"]].sum(axis=1)

    df_m = {}
    for m, m_short in MODEL_SHORTS.items():
        if m not in df["model"].values:
            continue
        df_m[m_short] = df[df["model"] == m].copy()
    df_gc = df_m["gc"]

    record_id_split = df_gc["record_id"].str.split("_", expand=True)
    df_gc["source"], df_gc["ibtracs_id"] = record_id_split[0], record_id_split[1]
    df_gc["year"] = df_gc["ibtracs_id"].str.slice(0, 4).astype(int)
    n_extents = df_gc.shape[0]
    n_storms = np.unique(df_gc["ibtracs_id"]).size
    period = (df_gc["year"].min(), df_gc["year"].max())

    dry_area_p = (df_gc["fp"] + df_gc["tn"]).sum() / df_gc["total_area"].sum()
    model_wet_area_p = df_gc["tp"].sum() / (df_gc["fp"] + df_gc["tp"]).sum()
    df_gc["obs_wet"] = df_gc["tp"] + df_gc["fn"]
    obs_wet_total = df_gc["obs_wet"].sum()
    df_gc["obs_wet_p"] = df_gc["obs_wet"] / obs_wet_total
    obs_wet_cum = df_gc.sort_values(by="obs_wet", ascending=False)["obs_wet"].cumsum()
    obs_wet_nhalf = (obs_wet_cum / obs_wet_total < 0.5).sum() + 1
    obs_wet_nsmall = (df_gc["obs_wet"] < obs_wet_total / n_extents).sum()
    obs_wet_nxsmall = (df_gc["obs_wet_p"] < 0.0005).sum()

    print_stats(
        "Coastal area",
        {
            "fm": df_gc["total_area"],
        },
        total=True,
    )

    print_stats(
        "Flooded area according to ...",
        {
            "fm": df_gc["tp"] + df_gc["fn"],
            **{
                k: d["tp"] + d["fp"]
                for k, d in df_m.items()
            }
        },
        total=True,
    )

    print(
        "Number of extents by region: "
        + ", ".join([
            f"{(df_gc['region'] == region_name).sum()} ({region_name})"
            for region_name in np.unique([n for n, _ in world_regions])
        ])
    )

    print(
        f"{n_extents} flood extents for {n_storms} storms in the period {period[0]}-{period[1]}."
        f" {100 * dry_area_p:.1f}% of the coastal area is dry according to satellite flood maps."
        f" {100 * model_wet_area_p:.1f}% of modeled wet areas are wet in observations."
    )

    print(
        f"The {obs_wet_nhalf} largest events together account for more than 50% of the areas"
        f" observed as wet. {obs_wet_nsmall} events are smaller than the average area."
        f" {obs_wet_nxsmall} events are smaller than 0.5% of the area."
    )


def print_stats_hwms(df):
    df_m = {}
    for m, m_short in MODEL_SHORTS.items():
        if m not in df["model"].values:
            continue
        df_m[m_short] = df[df["model"] == m].copy()
    df_gc = df_m["gc"]

    n_hwms = np.unique(df_gc["record_id"]).size
    n_storms = np.unique(df_gc["record_id"].str.split("_", expand=True)[0]).size

    print(
        "Share of HWMs with height above ground of less than one meter:",
        ((df_gc["hwm_above_gnd_m"] > 0) & (df_gc["hwm_above_gnd_m"] < 1.0)).sum()
        / ((df_gc["hwm_above_gnd_m"] > 0)).sum()
    )

    print_stats(
        "Height above ground according to ...",
        {
            "hwm": df_gc.loc[df_gc["hwm_above_gnd_m"] > 0, "hwm_above_gnd_m"],
            **{
                k: d.loc[d["model_above_gnd_m"] > 0, "model_above_gnd_m"]
                for k, d in df_m.items()
            }
        },
    )

    print(f"{n_hwms} HWMs for {n_storms} storms.")

    if "dem" not in df["model"].values:
        return

    df_dem = df[(df["model"] == "dem") & np.isfinite(df["dinund_signed"])].copy()
    n_hwms_topo = df_dem.shape[0]
    n_storms_topo = np.unique(df_dem["record_id"].str.split("_", expand=True)[0]).size
    n_impossible = df_dem["dtopo_gt_inund"].sum()
    dtopo_min, dtopo_max = (df_dem["dinund_signed"].min(), df_dem["dinund_signed"].max())
    print(
        f"{n_hwms_topo} HWMs with topo info ({n_storms_topo} storms)."
        f" Deviation from {dtopo_min:+.2f} to {dtopo_max:+.2f} meters."
        f" For {n_impossible} HWMs, the DEM's overestimation of topo heights is higher than"
        f" the actual flooding."
    )


def _compute_mcc_score(tp, tn, fp, fn):
    """Matthews Correlation Coefficient (MCC)"""
    return (tp * tn - fp * fn) / np.fmax(np.spacing(1),
        np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
    )


def _compute_tnr_score(tp, tn, fp, fn):
    """True negative hit rate"""
    return tn / (tn + fp)


def _compute_f1_score(tp, tn, fp, fn):
    """F1-score from Cea et al. 2022"""
    # hit rate (recall)
    hr = tp / (tp + fn)
    # false alarm ratio (1 - precision)
    far = fp / np.fmax(np.spacing(1), tp + fp)
    return (2 * hr * (1 - far)) / np.fmax(np.spacing(1), hr + (1 - far))


def _compute_f2_score(tp, tn, fp, fn):
    """F2-score, also known as critical success index (CSI)"""
    return tp / (fn + tp + fp)


def _compute_bias_score(tp, tn, fp, fn):
    """Frequency bias (see Stephenson 2000)"""
    # we log-transform the bias only after aggregation to avoid -inf values during aggregation
    return (tp + fp) / (tp + fn)


def stats_extents(df, *group_cols, metrics=None):
    metrics = ["tnr", "mcc", "f1", "f2", "bias"] if metrics is None else metrics

    df_single = df[[*group_cols, "model", "tp", "tn", "fp", "fn", "record_id"]]

    df_agg = (
        df_single
        .groupby(by=["model", *group_cols])
        [["tp", "tn", "fp", "fn"]]
        .sum()
        .reset_index()
    )
    df_agg["record_id"] = "*"

    df_scores = []
    for is_agg, df in zip([True, False], [df_agg, df_single]):
        args = (df["tp"], df["tn"], df["fp"], df["fn"])
        for m in metrics:
            df[m] = {
                "tnr": _compute_tnr_score,
                "mcc": _compute_mcc_score,
                "f1": _compute_f1_score,
                "f2": _compute_f2_score,
                "bias": _compute_bias_score,
            }[m](*args)
        df = df.melt(
            id_vars=["model", "record_id", *group_cols],
            value_vars=metrics,
            var_name="indicator",
            value_name="value",
        )
        if is_agg:
            df = (
                df
                .drop(columns=["record_id"])
                .rename(columns={"value": "total"})
            )
        else:
            df["is_finite"] = np.isfinite(df["value"])
            df = (
                df
                .groupby(["model", "indicator", *group_cols])
                .apply(
                    lambda v: pd.Series({
                        "N": v["is_finite"].sum(),
                        "mean": v["value"].where(v["is_finite"]).mean(),
                        "median": v["value"].where(v["is_finite"]).median(),
                        "17": v["value"].where(v["is_finite"]).quantile(q=0.17),
                        "83": v["value"].where(v["is_finite"]).quantile(q=0.83),
                    })
                )
                .reset_index()
            )
        df_scores.append(df)
    df = pd.merge(
        df_scores[0], df_scores[1], on=["model", "indicator", *group_cols], how="outer",
    )

    # log-transform bias score
    cols = ["total", "mean", "median", "17", "83"]
    mask = (df["indicator"] == "bias")
    df.loc[mask, cols] = np.log(np.fmax(np.spacing(1), df.loc[mask, cols]))

    return df


def stats_hwms_filtered(source, fes, cama_prot, models, panel_config, riverine=False):
    h_above_grnd, exclude_zeros, exclude_by_dtopo = panel_config
    df = data_hwms(
        source, fes, cama_prot,
        h_above_gnd=h_above_grnd,
        exclude_zeros=exclude_zeros,
        exclude_by_dtopo=exclude_by_dtopo,
        riverine=riverine,
    ).drop(columns=["dtopo", "dtopo_signed"])

    # restrict to a selection of models
    df = df[df["model"].isin(models)].copy()

    if exclude_by_dtopo:
        print(
            "Number of HWMs after DEM filtering:",
            np.isfinite(df.loc[df["model"] == "geoclaw", "dinund"].astype(float)).sum(),
        )
    return (
        df
        .melt(
            id_vars=["model", "record_id"],
            value_vars=["dinund", "dinund_signed"],
            var_name="indicator",
            value_name="value",
        )
        .groupby(["model", "indicator"])
        .apply(
            lambda v: pd.Series({
                "N": np.isfinite(v["value"].astype(float)).sum(),
                "mean": v["value"].mean(),
                "median": v["value"].median(),
                "17": v["value"].quantile(q=0.17),
                "83": v["value"].quantile(q=0.83),
            })
        )
        .reset_index()
    )


def stats_hwms(source, fes, cama_prot, models, riverine=False):
    df = data_hwms(source, fes, cama_prot, riverine=riverine)

    # treat DEM comparison like flood model comparisons
    df_dem = df[df["model"] == "geoclaw"].copy()
    df_dem["model"] = "dem"
    df_dem["model_flooded"] = True
    df_dem["dinund"] = df_dem["dtopo"]
    df_dem["dinund_signed"] = df_dem["dtopo_signed"]
    df = pd.concat([df, df_dem]).drop(columns=["dtopo", "dtopo_signed"])

    # restrict to a selection of models
    df = df[df["model"].isin(models)].copy()

    metrics = ["dinund", "dinund_signed", "model_flooded"]

    print_stats_hwms(df)
    return (
        df
        .melt(
            id_vars=["model", "record_id"],
            value_vars=metrics,
            var_name="indicator",
            value_name="value",
        )
        .groupby(["model", "indicator"])
        .apply(
            lambda v: pd.Series({
                "N": np.isfinite(v["value"].astype(float)).sum(),
                "mean": v["value"].mean(),
                "median": v["value"].median(),
                "17": v["value"].quantile(q=0.17),
                "83": v["value"].quantile(q=0.83),
            })
        )
        .reset_index()
    )
