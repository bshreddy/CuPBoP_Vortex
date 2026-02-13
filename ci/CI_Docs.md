# CuPBoP-Vortex CI/CD Infrastructure

This repository utilizes a distributed CI/CD pipeline that bridges GitHub Actions with the CRNCH (HPC) Cluster. It is designed to run Vortex simulations using Slurm and Apptainer.

## Architecture Overview
To do's: CI overview, folder structure, future updates, recompiling tools (LLVM, Vortex)
A job is only marked as Passed in GitHub if the Slurm job exits with code 0 AND the last 10 lines of the log file contain the following markers:

    --- Kernel compilation completed!

    --- Host compilation completed!

    --- Execution completed!

## Manual Testing & Reproduction

If a CI job fails, maintainers should reproduce the environment manually on a compute node. Do not run tests on the login node.

1. Start an Interactive Session

Request a node with the same specs used by the CI:

srun -p rg-nextgen-hpc -w dash[1-4] -N 1 --cpus-per-task=4 --mem=32G --gres=gpu:1 -t 00:30:00 --pty /bin/bash

2. Enter the Apptainer Environment and set environment variables

The CI environment is immutable and stored in a .sif file. Enter it using:
Bash

apptainer shell --nv -B /projects/ci-runners/CuPBoP-Vortex/:/test cupbop_env.sif
source ~/projects/ci-runners/CuPBoP-Vortex/temp-runner/_work/CuPBoP_Vortex/CuPBoP_Vortex/ci/rg-ci-setup.sh
export VORTEX_SCHEDULE_FLAG=0 # or 2 if testing different scheduler algorithm

3. Build CuPBoP-Vortex

cd ~/projects/ci-runners/CuPBoP-Vortex/temp-runner/_work/CuPBoP_Vortex/CuPBoP_Vortex/
rm -rf build; mkdir build; cd build
cmake .. -DLLVM_CONFIG_PATH=`which llvm-config` # need path to llvm-config
make

4. Executing Tests:
cd ~/projects/ci-runners/CuPBoP-Vortex/temp-runner/_work/CuPBoP_Vortex/CuPBoP_Vortex/examples/YOUR_EXAMPLE
./kjrun_llvm18.sh
