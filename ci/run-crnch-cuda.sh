#!/bin/bash
#SBATCH --job-name="CuPBoP-Vortex-ci-cuda"
#SBATCH -t 00:10:00                              # Duration of the job
#SBATCH -p rg-nextgen-hpc
#SBATCH -w dash3
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /tools/ci-reports/CuPBoP_logs/CuPBoP-cuda-test-%j.out   # Combined output and error messages file
#SBATCH -W                                       # Do not exit until the submitted job terminates.

##Add commands here to build and execute
cd $GITHUB_WORKSPACE
hostname

rm -rf build

apptainer exec --nv -B /projects/ci-runners/CuPBoP-Vortex/:/projects/ci-runners/CuPBoP-Vortex/ /projects/ci-runners/CuPBoP-Vortex/tools/cupbop_env.sif /bin/bash << 'EOF'
    source "./ci/rg-ci-setup.sh"
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
EOF
