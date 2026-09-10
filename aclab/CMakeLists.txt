cmake_minimum_required(VERSION 3.16.0)
project(aclab CXX)

set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann-9.1.0/aarch64-linux)
set(SOC_VERSION ascend910b3)
set(ASCEND_KERNEL_LAUNCH_ONLY ON)

include(/usr/local/Ascend/cann-9.1.0/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

ascendc_library(k2dev STATIC hello_kernel.cpp)
