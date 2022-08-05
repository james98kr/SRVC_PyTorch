#!/bin/bash

# This file decodes the 480x270 H.265-encoded mp4 files into 
# low-resolution(LR) YUV420p raw files.

encode_dir=$1
decode_dir=$2

for file in ${encode_dir}/*;
do
    onlyname=$(basename ${file})
    onlyname=${onlyname%.*}_dec
    
    outputname=${decode_dir}/${onlyname}.yuv

    ffmpeg -y -i ${file} -c:v rawvideo -pix_fmt yuv420p -vsync 0 ${outputname}
done