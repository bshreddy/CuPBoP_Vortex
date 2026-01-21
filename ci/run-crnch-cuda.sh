#!/bin/bash
#SBATCH --job-name="CuPBoP-Vortex-ci-cuda"
#SBATCH -t 00:10:00                              # Duration of the job
#SBATCH -w dash[1-4]
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -o /tools/ci-reports/CuPBoP_logs/CuPBoP-cuda-test-%j.out   # Combined output and error messages file
#SBATCH -W                                       # Do not exit until the submitted job terminates.

##Add commands here to build and execute
cd $GITHUB_WORKSPACE
hostname

#module use /projects/tools/x86_64/ubuntu-22.04/nvhpc-24.9/modulefiles/
#module load nvhpc/24.9
# Installing prebuilt Vortex package
# rm -rf vortex
rm -rf build
source "./ci/rg-ci-env-setup.sh"
#export CuPBoP_PATH=$PWD
#bash $CuPBoP_PATH/ci/toolchain_install.sh --vortex

# Installing prebuilt Vortex Toolchain
# cd vortex
# mkdir tools
# export TOOLDIR=$CuPBoP_PATH/vortex/tools # <== make tooldir visible to child processes (i.e. bash scripts)
# bash $CuPBoP_PATH/ci/toolchain_install.sh --all #NOTE <-- this will have to be changed to use the correct version of LLVM
# cd build
# ../configure --xlen=64 --tooldir=$TOOLDIR

# Installing CUDA headers for CuPBoP
cd $CuPBoP_PATH
wget "https://dl.dropboxusercontent.com/scl/fi/m9ap1tiybau4zk720t2z7/cuda-header.tar.gz?rlkey=zmdpst5l66t48ywrbtkj426nu&st=luao6zy7" -O cuda-header.tar.gz
tar -xzf 'cuda-header.tar.gz'
cp -r include/* runtime/threadPool/include/
rm -r 'cuda-header.tar.gz'

mkdir -p $CuPBoP_PATH/build && cd $CuPBoP_PATH/build
# export LLVM_CONFIG_PATH=$LLVM_VORTEX/bin/llvm-config
cmake .. -DLLVM_CONFIG_PATH=$LLVM_CONFIG_PATH
make
