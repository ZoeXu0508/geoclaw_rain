
# Tropical cyclone storm surge model validation

Validate outputs of the GeoClaw-based tropical cyclone storm surge model.

This Python package provides scripts to obtain, pre-process, and evaluate observational data of tropical cyclone-induced
storm surges and their flood maps, and compare the observational data to model outputs. The focus is on outputs from
a GeoClaw-based tropical cyclone storm surge model, but other models are considered, as well: the Global Tide and
Surge Model (GTSM), and three "bathtub" approaches to generate flood maps.

Since the main product of interest are gridded flood maps, the validation starts from an analysis of the main sources
for observational flood map products, and goes on to identify complementary products such as field (high water marks)
and tide gauge measurements to understand the sources of model biases.

## Installation

It is recommended to set up a conda environment with GDAL and NetCDF libraries. The code in this repository is
structured as a Python package called `gcvalid`. To install that package, run the following command line in the root
directory of this repository:

```shell
(env) $ pip install -e .
```

While most input files are downloaded on the fly, some files need to be placed manually in the right locations because
they are not publicly available, or access is restricted so that automatic retrieval is not possible:

1. Download the GESLA3 data following the [official instructions](https://gesla787883612.wordpress.com/downloads/).
In particular, store the [CSV metadata file](https://www.icloud.com/iclouddrive/01a8u37HiumNKbg6CpQUEA7-A#GESLA3_ALL_2)
in `input/gauges/gesla3_data/GESLA3_ALL.csv` and
the [data ZIP file](https://www.icloud.com/iclouddrive/0tHXOLCgBBjgmpHecFsfBXLag#GESLA3)
in `input/gauges/gesla3_data/GESLA3.0_ALL.zip` (do not unzip!).

2. The CoDEC data is expected to be placed in the `input/gauges/codec_data/` directory, including a
file `coor_coastal.nc` with an index of all stations, and directories `cf_tides` and `cf_esl` that contain one NetCDF
file per virtual tide gauge station.

3. For GESLA2 and GTSR data, the preprocessed input and hourly output data files from
the [gslcomp](https://gitlab.pik-potsdam.de/sitreu/gslcomp.git) project are used. For that, place the gslcomp root
directory in `input/gauges/gslcomp/`.

4. As elevation data set, an overlay of several topography and bathymetry data sets is used. Instructions on how to
generate this data set are provided in a
separate [git repository](https://gitlab.pik-potsdam.de/tovogt/combined-coastal-dem). Place the result in
`input/elevation/dem/combine.vrt`.

5. Download the [Copernicus Global Surface Water](https://global-surface-water.appspot.com/download) data set using the
"download_water_data" tool (from PyPI). Store the GeoTIFF files in `input/water/occurrence/` and create VRT file called
`input/water/occurrence/occurrence.vrt`.

6. Download the [monthly satellite altimetry data set from CMEMS](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description)
and convert the heights above geoid to EGM96 (the original reference geoid is GOCO05s). Store the data
in `input/water/monthly_zos_aviso.nc`. Do the same for
the [monthly global ocean reanalysis data set](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
and store the file in `input/water/monthly_zos_mercator.nc`. In addition to that, create a copy of the satellite
altimetry data set with all values replaced by 0 in `input/water/monthly_zos_0.nc`.

## Running the pipe line

The complete pipeline from data retrieval, pre-processing, comparison, and post-processing of compare results is very
long. We split it up into a sequence of scripts that need to be executed in order.

The following command lines download and pre-process the observational data:

```shell
(env) $ python -m gcvalid.prep.download.dfo
(env) $ python -m gcvalid.prep.download.rapid
(env) $ python -m gcvalid.prep.download.gfd
(env) $ python -m gcvalid.prep.download.fev
(env) $ python -m gcvalid.prep.download.uhslc
(env) $ python -m gcvalid.prep.maps.clean_dfo
(env) $ python -m gcvalid.prep.maps.clean_rapid
(env) $ python -m gcvalid.prep.maps.clean_gfd
(env) $ for src in dfo rapid gfd; do \
            python -m gcvalid.prep.maps.meta $src; \
            python -m gcvalid.prep.maps.tracks $src; \
            python -m gcvalid.prep.maps.winds $src; \
            python -m gcvalid.prep.maps.elevation $src; \
            python -m gcvalid.prep.gauge.by_map $src; \
            python -m gcvalid.prep.gauge.winds $src; \
        done
(env) $ python -m gcvalid.prep.gauge.annual_mean_altimetry aviso
(env) $ python -m gcvalid.prep.gauge.annual_mean_altimetry mercator
(env) $ python -m gcvalid.prep.gauge.annual_mean_altimetry 0
```

The following command lines generate and post-process the GeoClaw model outputs:

```shell
(env) $ for src in dfo rapid gfd; do \
            python -m gcvalid.prep.gauge.gc_locations $src; \
            python -m gcvalid.prep.gc.jobs $src; \
        done
(env) $
(env) $ # run all the GeoClaw jobs listed in input/geoclaw/*/jobs/*.txt
(env) $ # copy the GeoClaw output files (*.tif and *-gauge_data.pickle) to input/geoclaw/*/results/
(env) $ # copy the GeoClaw config (*.data) files to input/geoclaw/*/meta/
(env) $
(env) $ for src in dfo rapid gfd; do \
            python -m gcvalid.prep.gc.read_sl $src; \
        done
```

The following command lines generate and post-process the bathtub model outputs:

```shell
(env) $ for src in dfo rapid gfd; do \
            python -m gcvalid.prep.maps.bathtub_climada $src; \
            python -m gcvalid.prep.maps.bathtub_input $src codec; \
            python -m gcvalid.prep.maps.bathtub_input $src geoclaw-fes_no; \
            python -m gcvalid.prep.maps.bathtub_input $src geoclaw-fes_min; \
            python -m gcvalid.prep.maps.bathtub_input $src geoclaw-fes_mean; \
            python -m gcvalid.prep.maps.bathtub_input $src geoclaw-fes_max; \
        done
(env) $
(env) $ # run the aqueduct bathtub tool with the data and configs in input/bathtub/input/
(env) $ # and write the output to input/bathtub/*/
(env) $
(env) $ for src in dfo rapid gfd; do \
            python -m gcvalid.prep.maps.bathtub_aq_output $src codec; \
            python -m gcvalid.prep.maps.bathtub_aq_output $src geoclaw-fes_no; \
            python -m gcvalid.prep.maps.bathtub_aq_output $src geoclaw-fes_min; \
            python -m gcvalid.prep.maps.bathtub_aq_output $src geoclaw-fes_mean; \
            python -m gcvalid.prep.maps.bathtub_aq_output $src geoclaw-fes_max; \
        done
```

Finally, the following command lines do the actual comparison:

```shell
(env) $ for src in dfo rapid gfd; do \
            python -m gcvalid.compare.maps $src --zos aviso-fes_max --pluvial bt_aq_codec; \
            python -m gcvalid.compare.maps $src --zos aviso-fes_max --pluvial bt_climada; \
            for fes in no min mean max; do \
                python -m gcvalid.compare.maps $src --zos aviso-fes_max --pluvial bt_aq_geoclaw-fes_${fes}; \
                for zos in aviso mercator 0; do \
                    python -m gcvalid.compare.maps $src --zos ${zos}-fes_${fes} --pluvial without; \
                done; \
            done; \
            python -m gcvalid.compare.gauge $src; \
        done
```

After that, you can open any of the notebooks in the `./notebooks/` directory in a Jupyter Lab instance. Make sure that
you have support for interactive widgets (`ipywidget`) and plots (`ipympl`).

## Sources of observational flood maps

We use three sources of observational flood maps, all of which are extracted from various satellite products.

### Dartmouth Flood Observatory (DFO)

The [Dartmouth Flood Observatory](http://floodobservatory.colorado.edu/) collects information about variations in
global surface water since 1993. Since 2010, the archive is maintained by the Institute of Arctic and Alpine
Research (INSTAAR) at the University of Colorado.

Apart from a [flood archive index file](http://floodobservatory.colorado.edu/Version3/FloodArchive.xlsx) that lists
global flood events together with a limited amount of meta data, the web site also provides flood map snap shots for
selected events. For this study, we only consider data sets that specifically mention TCs (hurricanes, typhoons etc.)
in the official DFO description or meta data.

We manually georeferenced the RGB image data. The JPEG-files as well as the ground control points used during
georeferencing are provided with this repository in `./input/floodmaps/dfo/images_clean/`
and `./input/floodmaps/dfo/images_gcps/`.

### RAdar-Produced Inundation Diary (RAPID)

Yang et al. (2021): A High-Resolution Flood Inundation Archive (2016–the Present) from Sentinel-1 SAR Imagery over
CONUS. Bulletin of the American Meteorological Society 102(5): E1064–E1079. https://doi.org/10.1175/BAMS-D-19-0319.1

### Global Flood Database (GFD)

Tellman et al. (2021): Satellite imaging reveals increased proportion of population exposed to floods.
Nature 596(7870): 80–86. https://doi.org/10.1038/s41586-021-03695-w


## Sources of observational tide gauge records

We considered several sources of tide gauge records but use only the GESLA3 data for our final analysis.

### GESLA3

tba

### GESLA2

tba

### World Sea Levels (WSL)

tba (by the EU Joint Research Center, JRC)

### University of Hawaii Sea Level Center (UHSLC)

tba


## Source of observational high water marks (HWMs)

We obtain HWMs from field measurements of the US Geological Survey (USGS), as available from the Flood Event
Viewer (FEV) web site.
