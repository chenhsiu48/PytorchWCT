#!/bin/bash

for s in oil1 pencil2 water2 ink4; do
    ./WCT.py --style style/$s.jpg input/video/frames/*.png
    /usr/bin/ffmpeg -y -framerate 60 -i output/ntu%04d-$s.png -c:v h264_nvenc -b:v 1400k -vf 'fps=60,format=yuv420p,tmix=frames=3:weights="1 2 1"' -aspect 16:9 output/ntu-$s.mp4
    #rm output/ntu*-$s.png
done
