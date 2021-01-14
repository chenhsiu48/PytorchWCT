
.PHONY: clean

image:
	./WCT.py --content input/woman.jpg --style style/ink6.jpg
	./WCT.py --content input/woman.jpg --style style/oil6.jpg
	./WCT.py --content input/woman.jpg --style style/pencil2.jpg
	./WCT.py --content input/woman.jpg --style style/water.jpg
	./WCT.py --content input/lake.jpg --style style/ink6.jpg
	./WCT.py --content input/lake.jpg --style style/oil5.jpg
	./WCT.py --content input/lake.jpg --style style/pencil2.jpg
	./WCT.py --content input/lake.jpg --style style/water3.jpg
	./WCT.py --content input/street.jpg --style style/ink4.jpg
	./WCT.py --content input/street.jpg --style style/oil2.jpg

video:
	./WCT.py --style style/oil1.jpg input/ntu/*.png --gamma 0.95 --delta 0.6 --no_saliency
	/usr/bin/ffmpeg -y -framerate 60 -i output/ntu%04d-oil1.png -c:v h264_nvenc -b:v 5000k -vf 'fps=60,format=yuv420p,tmix=frames=3:weights="1 2 1"' -aspect 16:9 output/ntu-oil1.mp4
	./WCT.py --style style/pencil2.jpg input/ntu/*.png --gamma 0.95 --delta 0.6 --no_saliency
	/usr/bin/ffmpeg -y -framerate 60 -i output/ntu%04d-pencil2.png -c:v h264_nvenc -b:v 5000k -vf 'fps=60,format=yuv420p,tmix=frames=3:weights="1 2 1"' -aspect 16:9 output/ntu-pencil2.mp4
	./WCT.py --style style/water2.jpg input/ntu/*.png --gamma 0.95 --delta 0.6 --no_saliency
	/usr/bin/ffmpeg -y -framerate 60 -i output/ntu%04d-water2.png -c:v h264_nvenc -b:v 5000k -vf 'fps=60,format=yuv420p,tmix=frames=3:weights="1 2 1"' -aspect 16:9 output/ntu-water2.mp4
	./WCT.py --style style/ink6.jpg input/ntu/*.png --gamma 0.95 --delta 0.6 --no_saliency
	/usr/bin/ffmpeg -y -framerate 60 -i output/ntu%04d-ink6.png -c:v h264_nvenc -b:v 5000k -vf 'fps=60,format=yuv420p,tmix=frames=3:weights="1 2 1"' -aspect 16:9 output/ntu-ink6.mp4

release:
	mkdir -p release release/style release/input release/output
	cp -r requirements.txt Makefile Readme.md *.py models doc release
	cp style/*.jpg release/style; cp input/woman.jpg input/lake.jpg input/street.jpg release/input; cp -r input/video/frames release/input/ntu
	cp samples/*.png release/output; cp samples/*.mp4 release/output
	cd release; zip -r ../DIP2020-G12-Style-Transfer.zip *

clean:
	@rm -rf release *.zip
