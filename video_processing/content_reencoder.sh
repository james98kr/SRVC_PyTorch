#!/bin/bash

# This file takes the 1080p YUV420p raw format video files encode them using H.265
# without downsampling or resizing. The output will be used as the 1080p H.264/H.265
# baseline for comparison. 

yuv_dir=$1
output_dir=$2
original_dir=$3

for file in ${yuv_dir}/*;
do
    onlyname=$(basename ${file})
    onlyname=${onlyname%.*}
    oheight=1080
    owidth=1920
    height=1080
    width=1920
    
    originalname=${original_dir}/${onlyname}*
    fps=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 \
        -show_entries stream=r_frame_rate ${originalname})

    for format in 265;
    do
        for crf in 10 20 30 40 50;
        do
            outputname=${output_dir}/${onlyname}_crf${crf}.mp4
            ffmpeg -y -f rawvideo -s ${owidth}x${oheight} -pix_fmt yuv420p -framerate ${fps} \
                -i ${file} -vf scale=${width}:${height}:out_color_matrix=bt709 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range 1 \
                -sws_flags area -vcodec libx${format} -preset slow \
                -vsync 0 -pix_fmt yuv420p \
                -crf ${crf} -an ${outputname}
        done
    done
done