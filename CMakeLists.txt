cmake_minimum_required(VERSION 3.10)

project(Samples VERSION 0.5)

set(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")


if(DEFINED ENV{CUDA_PATH})
link_directories($ENV{CUDA_PATH}/lib64)
else()
link_directories(/usr/local/cuda/lib64)
endif()

if(DEFINED ENV{CUDNN_PATH})
link_directories($ENV{CUDNN_PATH}/lib64)
endif()

add_executable(Samples samples/conv_sample.cpp 
                       samples/test_list.cpp 
                       samples/fp16_emu.cpp 
                       samples/catch.cpp 
                       samples/helpers.cpp 
                       samples/fusion_sample.cpp 
                       samples/mha_sample.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

if(DEFINED ENV{CUDA_PATH})
    target_include_directories(Samples PUBLIC $ENV{CUDA_PATH}/include/)
    target_include_directories(Samples PUBLIC $ENV{CUDA_PATH}/targets/ppc64le-linux/include)
else()
    target_include_directories(Samples PUBLIC /usr/local/cuda/include/)
    target_include_directories(Samples PUBLIC /usr/local/cuda/targets/ppc64le-linux/include)
endif()

if(DEFINED ENV{CUDNN_PATH})
    target_include_directories(Samples PUBLIC $ENV{CUDNN_PATH}/include/)
endif()

if(DEFINED ENV{CUDNN_FRONTEND_PATH})
    target_include_directories(Samples PUBLIC $ENV{CUDNN_FRONTEND_PATH}/include/)
    target_include_directories(Samples PUBLIC $ENV{CUDNN_FRONTEND_PATH}/contrib/catch2/)
else()
    target_include_directories(Samples PUBLIC include/)
    target_include_directories(Samples PUBLIC contrib/catch2/)
endif()

target_link_libraries(Samples cudart)
target_link_libraries(Samples cudnn)
target_link_libraries(Samples m)
