#!/bin/bash

# -s 1920x1080 \
cd ../data && \
ffpb \
-pattern_type glob \
-y \
-r 10 \
-i "proj_*.png" \
-vcodec libx264 \
-crf 25 \
-pix_fmt yuv420p \
-hide_banner \
../videos/proj_0$1.mp4 # -loglevel error \

cd ../videos && \
ffmpeg \
    -i proj_0.mp4 \
    -vf "fps=10,scale=1920:-1:flags=lanczos" \
    -c:v gif \
    -f gif output.gif

# !ffmpeg -pattern_type glob -y -r 60 -i "a*.jpeg" -vcodec h264 -pix_fmt yuv420p a_output.mp4 -hide_banner -loglevel error
# !ffmpeg -pattern_type glob -y -r 60 -i "b*.jpeg" -vcodec h264 -pix_fmt yuv420p b_output.mp4 -hide_banner -loglevel error
# !ls | grep ".jpeg" | xargs rm