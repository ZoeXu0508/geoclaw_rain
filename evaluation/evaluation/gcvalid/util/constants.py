
import pathlib

import numpy as np

import gcvalid


BASE_DIR = pathlib.Path(gcvalid.__file__).resolve().parent.parent
"""Base directory for all paths defined below (usually the repository's root)

Note that the __file__ attribute points to the "gcvalid/__init__.py" file.
"""

OUTPUT_DIR = BASE_DIR / "output"
INPUT_DIR = BASE_DIR / "input"

BATHTUB_DIR = INPUT_DIR / "bathtub"
ELEVATION_DIR = INPUT_DIR / "elevation"
ELEVATION_MAPS_DIR = ELEVATION_DIR / "by_floodmap"
CATCHMENTS_DIR = ELEVATION_DIR / "catchments"
FLOODMAPS_DIR = INPUT_DIR / "floodmaps"
GAUGES_DIR = INPUT_DIR / "gauges"
GEOCLAW_DIR = INPUT_DIR / "geoclaw"
HWMS_DIR = INPUT_DIR / "hwms"
INDEX_DIR = INPUT_DIR / "index"
PLUVIAL_DIR = INPUT_DIR / "pluvial"
PLUVIAL_MAPS_DIR = PLUVIAL_DIR / "by_floodmap"
RAINFALL_MAPS_DIR = PLUVIAL_MAPS_DIR / "rainfall"
RAINFALL_RAW_DIR = PLUVIAL_DIR / "raw" / "rainfall"
RUNOFF_MAPS_DIR = PLUVIAL_MAPS_DIR / "runoff"
TRACKS_DIR = INPUT_DIR / "tracks"
WINDS_DIR = INPUT_DIR / "winds"
WATER_DIR = INPUT_DIR / "water"

COMPARE_DIR = OUTPUT_DIR / "compare"
SOURCEDATA_DIR = OUTPUT_DIR / "sourcedata"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

DEM_FILE = ELEVATION_DIR / "dem" / "combine.vrt"
WATERBODY_FILE = WATER_DIR / "occurrence" / "occurrence.vrt"
HWMS_FILE = HWMS_DIR / "flood_event_viewer.json"

CODEC_DIR = GAUGES_DIR / "codec_data"
CODEC_COORD_FILE = CODEC_DIR / "coor_coastal.nc"
CODEC_TIDES_DIR = CODEC_DIR / "cf_tides"
CODEC_COMBINED_DIR = CODEC_DIR / "cf_esl"

GSLCOMP_RUN_ID = "GSLcomp_09_cutoff_3"
GSLCOMP_DIR = GAUGES_DIR / "gslcomp"
GSLCOMP_INPUT = GSLCOMP_DIR / "input"
GSLCOMP_OUTPUT = GSLCOMP_DIR / "output"
GSLCOMP_GTSM_DIR = GSLCOMP_OUTPUT / GSLCOMP_RUN_ID / "hourly_time_series"
GSLCOMP_GESLA_DIR = GSLCOMP_INPUT / "GESLA_2_unique"
GSLCOMP_GTSM_INDEX_FILE = GSLCOMP_INPUT / "preprocessed_input" / 'gtsm_index.h5'
GSLCOMP_GESLA_INDEX_FILE = GSLCOMP_INPUT / "preprocessed_input" / "gesla2_lat_lon.h5"


WFDE5_PATH = RAINFALL_RAW_DIR / "WFDE5"
WFDE5_FNAME = "Rainf_WFDE5_CRU_{year:04d}{month:02d}_daily.nc"
"""Path to gridded daily WFDE5 precipitation data

The data has been aggregated from the hourly data in the following location:

    /p/projects/climate_data_central/observation/WFDE5/v2.0/Rainf_WFDE5_CRU/
"""


GPCC_PATH = RAINFALL_RAW_DIR / "gpcc"
GPCC_FNAME = "full_data_daily_v2020_10_{year}.nc"
"""Path to gridded daily GPCC precipitation data"""


ERA5L_PATH = RAINFALL_RAW_DIR /  "ERA5-Land"
ERA5L_PR_FNAME = "total_precipitation_ERA5-Land_{year:04d}{month:02d}_daily.nc"
ERA5L_SRO_FNAME = "surface_runoff_ERA5-Land_{year:04d}{month:02d}_daily.nc"
"""Path to gridded daily ERA5-Land precipitation and surface runoff data

The precipitation data has been aggregated from the hourly data in the following location:

    /p/projects/climate_data_central/reanalysis/ERA5-Land/total_precipitation/

The surface runoff data has been downloaded from Copernicus.
"""


ERA5_PATH = RAINFALL_RAW_DIR /  "ERA5-daily"
ERA5_PR_FNAME = "pr_daily_ECMWF-ERA5_observation_2000010100-2019123123.nc"
ERA5_SRO_FNAME = "sro_daily_ECMWF-ERA5_observation_1979010100-2021123123.nc"
"""Path to gridded daily ERA5 precipitation and surface runoff data

The precipitation data has been aggregated from the hourly data in the following location:

    /p/projects/climate_data_central/reanalysis/ERA5/pr/

The surface runoff data has been downloaded from Copernicus.
"""


DEFAULT_CRS = "epsg:4326"
"""All external raster data are transformed to this CRS for further processing."""


GAUGE_MODELS = [
    'codec',
    'geoclaw-zos_aviso-fes_no',
    'geoclaw-zos_aviso-fes_bestrmse',
    'geoclaw-zos_aviso-fes_max',
    'geoclaw-zos_aviso-fes_mean',
    'geoclaw-zos_aviso-fes_min',
]
GAUGE_MODELS_SHORT = [m.replace("-zos_aviso", "") for m in GAUGE_MODELS]
GAUGE_MODEL_SHORTNAMES = {m: ms for m, ms in zip(GAUGE_MODELS, GAUGE_MODELS_SHORT)}
GAUGE_MODEL_COLORS = {
    "observed": "black",
    "geoclaw-zos_aviso-fes_bestrmse": "pink",
    "geoclaw-zos_aviso-fes_min": "lightcoral",
    "geoclaw-zos_aviso-fes_mean": "indianred",
    "geoclaw-zos_aviso-fes_max": "maroon",
    "geoclaw-zos_aviso-fes_no": "violet",
    "codec": "tab:blue",
}
"""Models (as opposed to observed data) for the gauge analysis"""


GAUGE_SOURCE_MM_CONVERSION = {
    "gtsm": 1,
    "codec": 1,
    "gesla": 1000,
    "gesla3": 1000,
    "uhslc": 1,
    "wsl": 1000,
}
"""Conversion factors for conversion from original tide gauge data to mm units"""


SAFFIR_SIMPSON_THRESHS = [17, 32, 42, 49, 58, 70]
SAFFIR_SIMPSON_NAMES = ["TD", "TS"] + [f"C{i}" for i in range(1, 6)]
SAFFIR_SIMPSON_NAMES_LONG = (
    ["Tropical Depression", "Tropical Storm"] + [f"Category {i}" for i in range(1, 6)]
)
SAFFIR_SIMPSON_NAMES_SHORT = (
    ["Trop. Depression", "Trop. Storm"] + [f"Cat. {i}" for i in range(1, 6)]
)
SAFFIR_SIMPSON_MIN_BY_NAME = {
    n: t for n, t in zip(SAFFIR_SIMPSON_NAMES, [0] + SAFFIR_SIMPSON_THRESHS)
}
SAFFIR_SIMPSON_COLORS = [
    # color scale similar to Bloemendaal et al. (2020)
    "#5ebaff", "#00faf4", "#f0da70", "#fea230", "#f06013", "#e03323", "#bb0000",
]
SAFFIR_SIMPSON_YSCALE = list(zip(
    SAFFIR_SIMPSON_NAMES,
    SAFFIR_SIMPSON_COLORS,
    SAFFIR_SIMPSON_THRESHS + [100],
))
"""Thresholds, names and colors for working with the Saffir-Simpson hurricane scale."""

CM_TO_INCH = 1.0 / 2.54

PLOT_WIDTH_IN = 19 * CM_TO_INCH
"""The plot width in inches: 9cm (single column), 14cm (1.5 column), 19cm (double column)"""
