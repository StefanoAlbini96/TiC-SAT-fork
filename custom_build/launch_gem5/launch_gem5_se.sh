#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Invalid call. Usage: './launch_se.sh   dir_path    stats_file    config_file'" 1>&2
    exit 1
fi

/home/albini/Documents/ESL/TiC-SAT-fork/gem5/build/ARM/gem5.fast \
-d $1 \
--stats-file=$2.txt \
--dump-config=$3.txt \
/home/albini/Documents/ESL/TiC-SAT-fork/gem5/configs/example/arm/starter_se.py \
--cpu="minor" \
--num-cores=1 \
--sa-size=8
