#!/bin/bash

#SBATCH --qos=priority
#SBATCH --job-name=dfo_run_script
#SBATCH --account=ebm
#SBATCH --chdir=/home/tovogt/code/dfo_validation/log
#SBATCH --output=run_script-%j.out
#SBATCH --error=run_script-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

module load intel/2019.0-nopython

export CONDA_HOME=/home/tovogt/.local/share/miniforge3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $CONDA_HOME/etc/profile.d/conda.sh
conda activate isimip3a

cd /home/tovogt/code/dfo_validation
echo $@ > log/run_script-${SLURM_JOB_ID}.args
SCRIPT_NAME=$1
shift
srun python -u -m "gcvalid.$SCRIPT_NAME" $@
