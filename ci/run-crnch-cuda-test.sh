#!/bin/bash
#SBATCH --job-name="CuPBoP-Vortex-ci-cuda"
#SBATCH -t 00:30:00                              # Duration of the job
#SBATCH -p rg-nextgen-hpc
#SBATCH -w dash[1-4]
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /tools/ci-reports/CuPBoP_logs/CuPBoP-cuda-test-%j.out   # Combined output and error messages file
#SBATCH -W                                       # Do not exit until the submitted job terminates.

cd $GITHUB_WORKSPACE
hostname

export FINAL_OUTPUT_PATH="/tools/ci-reports/CuPBoP_logs/CuPBoP-cuda-test-${SLURM_JOB_ID}.out"
apptainer exec --nv -B /projects/ci-runners/CuPBoP-Vortex/:/projects/ci-runners/CuPBoP-Vortex/ /projects/ci-runners/CuPBoP-Vortex/tools/cupbop_env.sif /bin/bash << 'EOF'
    source "./ci/rg-ci-setup.sh"

    cd ./examples/${TEST_NAME}
    bash kjrun_llvm18.sh
EOF
