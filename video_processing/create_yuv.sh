#!/bin/bash

# This file takes original high-resolution(4K) mp4 files
# and changes them to 10-minute long videos in 1080p YUV420p raw format.

input_dir=$1
output_dir=$2

for file in ${input_dir}/*;
do
    onlyname=$(basename ${file})
    onlyname=${onlyname%.*}
    height=1080
    width=1920
    min=10
    
    outputname=${output_dir}/${onlyname}.yuv

    ffmpeg -y -i ${file} -ss 0 -t ${min}:00 \
        -vf scale=${width}:${height}:out_color_matrix=bt709 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range 1 \
        -c:v rawvideo -pix_fmt yuv420p -vsync 0 \
        ${outputname}
done