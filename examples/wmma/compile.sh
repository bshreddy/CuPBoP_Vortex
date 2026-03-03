#!/bin/bash

# exit when any command fails
set -e

######################### Default Varaibles #################################
DEVICE=vortex
KERNEL_CU=wmma.cu
ARCH=64
#############################################################################

# default: 1:1 mapping(2), can change it to thread mapping (0)
export VORTEX_SCHEDULE_FLAG=${VORTEX_SCHEDULE_FLAG:-2}

show_usage()
{
    echo "CuPBoP Compiling Script"
    echo "Usage: [--cuda_input=[cu filename]] [--device_target=[x86|simx|vortex] [--help]]"
}

if [ -z "$VORTEX_PATH" ]
then
    echo "You need to set "\$VORTEX_PATH" environment variable"
    exit -1
fi

if [ -z "$CuPBoP_PATH" ]
then
    echo "You need to set "\$CuPBoP_PATH" environment variable"
    exit -1
fi

if [ $ARCH = 32 ]
then
    RISCV_TOOLCHAIN_PREFIX=$RISCV_TOOLCHAIN/riscv32-unknown-elf-
    RISCV_TOOLCHAIN_FOLDER=$RISCV_TOOLCHAIN
elif [ $ARCH = 64 ]
then
    RISCV_TOOLCHAIN_PREFIX=$RISCV_TOOLCHAIN/riscv64-unknown-elf-
    RISCV_TOOLCHAIN_FOLDER=$RISCV_TOOLCHAIN
else
    echo "ARCH is setup to a wrong number, check your bash file"
    exit -1
fi

CUDA_PATH=$CuPBoP_PATH/cuda-12.1
DEBUG_LEVEL=0

DPRINT()
{
  if [ $DEBUG_LEVEL -eq 1 ]; then
      echo $@
  fi
}

for i in "$@"
do
case $i in
    --cuda_input=*)
        KERNEL_CU=${i#*=}
        shift
        ;;
    --device_target=*)
        DEVICE=${i#*=}
        shift
        ;;
    --help)
        show_usage
        exit 0
        ;;
    *)
    show_usage
    exit -1
    ;;
esac
done

case $DEVICE in
    x86)
        DRIVER_PATH=$VORTEX_PATH/driver/rtlsim
        ;;
    simx)
        DRIVER_PATH=$VORTEX_PATH/driver/simx
        ;;
    vortex)
        DRIVER_PATH=$VORTEX_PATH/driver/simx
        ;;
    *)
        echo "invalid device: $DEVICE"
        exit -1
        ;;
esac

# delete all generated files
rm -f *.out *.o *.dump *.log *.ll *.bc *.elf

KERNEL=`basename $KERNEL_CU .cu`
echo "--- Generate bitcode files(.bc) for host and device by using clang++"
# ${LLVM_PREFIX}/bin/clang++ -O0 -g -std=c++11  ./$KERNEL_CU -I../.. --sysroot=/ --target=x86_64-linux-gnu --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 --cuda-path=$CUDA_PATH --cuda-gpu-arch=sm_75 -L$CUDA_PATH/lib64  -lcudart_static -ldl -lrt -pthread -save-temps -v  || true
${LLVM_PREFIX}/bin/clang++ -O0 -g -std=c++11  ./$KERNEL_CU -I../.. --sysroot=/ --target=x86_64-linux-gnu --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 --cuda-path=$CUDA_PATH --cuda-gpu-arch=sm_75 -L$CUDA_PATH/lib64  -lcudart_static -ldl -lrt -pthread -save-temps || true

echo "--- Generate LLVM IR files(.ll) for host and device"
llvm-dis $KERNEL-cuda-nvptx64-nvidia-cuda-sm_75.bc
llvm-dis $KERNEL-host-x86_64-unknown-linux-gnu.bc

echo "--- Translate the kernel bitcode by using CuPBoP's kernel translator"
#$CuPBoP_PATH/build/compilation/kernelTranslator.x86 $KERNEL-cuda-nvptx64-nvidia-cuda-sm_50.bc kernel_host.bc
$CuPBoP_PATH/build/compilation/kernelTranslator $KERNEL-cuda-nvptx64-nvidia-cuda-sm_75.bc kernel.bc
llvm-dis kernel.bc
