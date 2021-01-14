#!/bin/bash

mkdir release

for s in oil1 pencil2 water2 ink6; do
    ./WCT.py --style style/$s.jpg input/video/frames/*.png --gamma 0.95 --delta 0.6 --no_saliency
    /usr/bin/ffmpeg -y -framerate 60 -i output/ntu%04d-$s.png -c:v h264_nvenc -b:v 5000k -vf 'fps=60,format=yuv420p,tmix=frames=3:weights="1 2 1"' -aspect 16:9 output/ntu-$s.mp4
    /usr/bin/ffmpeg -y -i input/video/ntu.mp4 -i output/ntu-$s.mp4 -c:v h264_nvenc -b:v 10000k -filter_complex hstack=inputs=2 c-ntu-$s.mp4
    rm output/ntu*-$s.png
done
