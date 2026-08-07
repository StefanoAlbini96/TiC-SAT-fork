cmake_minimum_required(VERSION 3.16)
project(FvllMontiTransformer)

set(CMAKE_CXX_STANDARD 17)

# set(GCC_COVERAGE_COMPILE_FLAGS "-DDEVELOP -DREARRANGE -DSA -DSA_SIZE=16 -DRELOAD_WEIGHT -DCORE_NUM=4 -fopenmp -O2")


set(GCC_COVERAGE_COMPILE_FLAGS
        "-DDEVELOP              \
        -DBWMA                  \
        -DREARRANGE             \
        -DSA                    \
        -DSA_SIZE=4             \
        -DCORE_NUM=1            \
        -fopenmp                \
                                \
        -DRELOAD_WEIGHT         \
                                \
        -DNANO_3D               \
        -DN_3D_LAYERS=2         \
        -DSA_W=4                \
        -DSA_H=8           \
        -O2")

        # -DNANO_3D               \
        # -DRELOAD_WEIGHT        \



set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

file(GLOB transformer_SRC
        "transformer_layers/*.h"
        "transformer_layers/*.cc"
        "accelerator/smm_gem.cpp"
        "accelerator/systolic_m2m.cc"
        )

add_executable(FvllMontiTransformer transformer.cpp ${transformer_SRC})
