#!/bin/bash
#SBATCH --job-name="CuPBoP-Vortex-ci-cuda"
#SBATCH -t 00:30:00                              # Duration of the job
#SBATCH -p rg-nextgen-hpc
#SBATCH -w dash[1-4]
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /tools/ci-reports/CuPBoP_logs/CuPBoP-cuda-test-%j.out   # Combined output and error messages file
#SBATCH -W                                       # Do not exit until the submitted job terminates.

##Add commands here to build and execute
cd $GITHUB_WORKSPACE
hostname

#module use /projects/tools/x86_64/ubuntu-22.04/nvhpc-24.9/modulefiles/
#module load nvhpc/24.9

export FINAL_OUTPUT_PATH="/tools/ci-reports/CuPBoP_logs/CuPBoP-cuda-test-${SLURM_JOB_ID}.out"
apptainer shell --nv -B /projects/ci-runners/CuPBoP-Vortex/:/projects/ci-runners/CuPBoP-Vortex/ /projects/ci-runners/CuPBoP-Vortex/tools/cupbop_env.sif
source "./ci/rg-ci-setup.sh"

cd ./examples/${TEST_NAME}
bash kjrun_llvm18.sh
