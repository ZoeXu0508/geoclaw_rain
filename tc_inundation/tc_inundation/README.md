
# Tropical cyclone flood maps from tracks, using GeoClaw

Compute tropical cyclone-induced storm surge and derive flood maps and tide gauge time series.

This set of scripts is built around the CLIMADA submodule `climada_petals.hazard.tc_surge_geoclaw`.
For more information about that feature, please refer to
the [CLIMADA docs](https://climada-petals.readthedocs.io/en/stable/tutorial/climada_hazard_TCSurgeGeoClaw.html).
This set of scripts implements an easy API to run the surge model for historical TC track data from IBTrACS for specific
regions of interest, with selected input data sets for topography, and base sea level.

## Installation

This emulator is tested on the PIK cluster (with SLURM). While the software dependencies can be set up on any
platform (e.g, Ubuntu, Windows, macOS) that is supported by the CLIMADA Python package, the computational requirements
are too high for most notebooks and personal workstation machines. It is recommended to launch a separate job for each
individual tropical cyclone, with 16 cores and 64 GB of memory.

To set up an environment for the scripts in this repository, install the Python project CLIMADA (including
climada_petals) in a conda environment following
the [advanced installation instructions](https://climada-python.readthedocs.io/en/stable/guide/install.html#advanced-instructions)
as specified in CLIMADA documentation. Make sure to check out the branch `feature/tc_surge_geoclaw` in climada_petals.
After that, only few additional dependencies have to be installed manually:

```shell
(climada_env) $ conda install gfortran_linux-64
(climada_env) $ conda install pyfes -c fbriol
```

The `pyfes` package is the Python interface to access the FES 2014 model outputs, as published in
the [GitHub repository by CNES and AVISO](https://github.com/CNES/aviso-fes).

## Input data

Before running any of the scripts, you need to obtain a number of input files.

The combined DEM product needs to be placed in `input/dem/combined/combine_v2.1.vrt`. If you only want to work with
SRTM15+, you only need `input/dem/srtm15plus/index.vrt`.

The monthly satellite altimetry data needs to be placed in `input/monthly_zos_<name>.nc`. Make sure that the data is
relative to the EGM96 geoid (according to the DEM data).

Download the ini-files from the [FES2014 git repository](https://github.com/CNES/aviso-fes/). Then, extract the
corresponding FES2014 model data as obtained from
`ftp://ftp-access.aviso.altimetry.fr/auxiliary/tide_model/fes2014_elevations_and_load/`:

```
input/fes2014/ocean_tide_extrapolated.ini
input/fes2014/ocean_tide_extrapolated/
input/fes2014/load_tide.ini
input/fes2014/load_tide/
```

## Basic execution

The heart of this project is the script `scripts/tc_run_geoclaw.py` which launches the GeoClaw solver for a single
tropical cyclone track in the IBTrACS database of historical storms. The script expects only a single input parameter
on the command line: the IBTrACS storm ID. However, in many applications, you will at least specify a region of interest
to reduce the computational requirements (using the `--bounds` parameter):

```shell
(climada_env) $ python scripts/tc_run_geoclaw.py 2003262N17254 --bounds B-122.07999,17.24871,-99.20202,36.85008
```

You can specify the location of virtual tide gauges to record water level time series during the simulation, or
change how base sea level is derived from satellite altimetry and astronomical tides. Use the `--help` parameter to
obtain a list of all supported parameters.

The main model outputs are GeoTIFF (`.tif`) files in subdirectories of the `output/` directory with the maximum surge
height over the storm life time for each grid cell in the region of interest.

## Execution on a SLURM cluster

You typically would not call the Python script manually from the command line as described above, but you would write
the command line arguments into a file and run a SLURM batch script. An example SLURM sbatch script that runs
`tc_run_geoclaw.py` (see previous section) on SLURM is included in `slurm/tc_run_geoclaw.sh.template`. Copy the template
to `slurm/tc_run_geoclaw.sh`, open the file in a text editor, and follow the instruction in the file to replace all of
the placeholders according to your environment.

The batch script is designed to run a whole job array where each job in the array executes `scripts/tc_run_geoclaw.py`
with command line parameters taken from a single line in a predefined job file. Hence, to run the script, you generate
a file `jobs/my_args.txt` with each line containing one set of arguments for which you would like to run the Python
script `scripts/tc_run_geoclaw.py`. Assuming that your job file has 10 lines, you submit the script to SLURM as follows:

```bash
(climada_env) $ sbatch --qos standby --array 1-10 slurm/tc_run_geoclaw.sh jobs/my_args.txt
```
The parameter `--qos standby` means that you may run the jobs in a "standby" queue, i.e., jobs may
be preempted at any time and will resume operation once they are restarted.

The log outputs of the batch jobs are written to files in the `log/` subdirectory, and there are several scripts to help
you extract more information from the log outputs. For example, `scripts/runtimes.py` will extract a job's run time,
and `scripts/store_meta.py` identifies the (internal) location of the GeoClaw run directory and copies central GeoClaw
configuration files to the `output/` directory. Finally, `scripts/check_failed_jobs.py` will list all jobs that ended
with an error message. It will also help you to understand the error messages, and restart the jobs if you think that
might help (e.g., in cases where you modified the code).
