#!/usr/bin/env bash
# =========================================================
#  Fast gem5 build script for ARM using GCC 13
# =========================================================
# Usage:
#   chmod +x build_gem5_arm_gcc13.sh
#   ./build_gem5_arm_gcc13.sh [opt|dbg|fast]
#
# Example:
#   ./build_gem5_arm_gcc13.sh opt
# =========================================================

# Stop on errors
set -e

# Go do the build directory
cd /home/albini/Documents/ESL/TiC-SAT-fork/gem5


# Choose build type: opt (optimized), dbg (debug), or fast (minimal optimization)
BUILD_TYPE=${1:-fast}

# Detect number of CPU cores for parallel build
JOBS=$(nproc)
# JOBS=1

# Path to gem5 root (adjust if running from another dir)
GEM5_ROOT="$(pwd)"

# Compiler configuration
export CC=gcc-13
export CXX=g++-13

# Target architecture
TARGET=ARM

echo "=========================================="
echo " Building gem5 for $TARGET ($BUILD_TYPE)"
echo " Using compiler: $($CXX --version | head -n 1)"
echo " Using $JOBS parallel jobs"
echo "=========================================="
sleep 1

# Clean previous build (optional)
# Uncomment this if you want a completely fresh build each time:
# scons -C "$GEM5_ROOT" -c build/$TARGET/gem5.$BUILD_TYPE

export LD_LIBRARY_PATH=/home/albini/miniconda3/lib:$LD_LIBRARY_PATH

# Build gem5
scons -j"$JOBS" -C "$GEM5_ROOT" build/$TARGET/gem5.$BUILD_TYPE --ignore-style


# Go back to the original directory 
cd -


echo "=========================================="
echo "✅ Build complete!"
echo "Output: build/$TARGET/gem5.$BUILD_TYPE"
echo "=========================================="


