# DIP2020 Group 12 Final Project

Our final project adopts the WCT method from the paper [Universal Style Transfer via Feature Transforms](https://arxiv.org/pdf/1705.08086.pdf). The official Torch implementation can be found [here](https://github.com/Yijunmaverick/UniversalStyleTransfer) and Tensorflow implementation can be found [here](https://github.com/eridgd/WCT-TF).
We extend [Daan Wynen](https://github.com/black-puppydog/PytorchWCT/blob/master/Readme.md)'s **modified** Pytorch implementation to build this final project to transfer various artistic styles on realistic photos. 

- Presentation slide: ```doc/G12_Style_Transfer_Slide.pdf```
- Final Report: ```doc/G12_Style_Transfer_Report.pdf```

## Prerequisites
- [Pytorch](http://pytorch.org/) 
- [FFmpeg](https://ffmpeg.org/): to encode the images to video

### Requirements
```
$ pip install -r requirements.txt
```

## Executation

To generate the style transferred images for ```woman.jpg```, ```lake.jpg```, and ```street.jpg```, simple execute: 
```
$ make image
```

The NTU video is dumped into raw PNG images under ```input/ntu/ntu%04d.png```. To generate the style transferred videos, simple execute: (it takes pretty long time)
```
$ make video
```

To apply different effects, invoking ```./WCT.py -h``` will list all the command line arguments to run our program. A simple way to transfer an image looks like: 

```
$ ./WCT.py --content input/woman.jpg --style style/ink6.jpg
```

or transfer a sequence of frames to be encoded as video: 

```
$ ./WCT.py --style style/oil1.jpg input/ntu/*.png --gamma 0.95 --delta 0.6 --no_saliency
```

Check the ```Makefile``` will give you more examples about how to execute this program. 

## Results

### Woman
![woman-oil](https://github.com/chenhsiu48/PytorchWCT/raw/master/samples/woman-oil6.png) ![woman-pencil](https://github.com/chenhsiu48/PytorchWCT/raw/master/samples/woman-pencil2.png)
![woman-ink](https://github.com/chenhsiu48/PytorchWCT/raw/master/samples/woman-ink6.png) ![woman-water](https://github.com/chenhsiu48/PytorchWCT/raw/master/samples/woman-water.png)

### Lake
![lake-oil](https://github.com/chenhsiu48/PytorchWCT/raw/master/samples/lake-oil5.png) ![lake-pencil](https://github.com/chenhsiu48/PytorchWCT/raw/master/samples/lake-pencil2.png) 
![lake-ink](https://github.com/chenhsiu48/PytorchWCT/raw/master/samples/lake-ink6.png) ![lake-water](https://github.com/chenhsiu48/PytorchWCT/raw/master/samples/lake-water3.png) 

### Street
![street-oil](https://github.com/chenhsiu48/PytorchWCT/raw/master/samples/street-oil2.png) 
![street-ink](https://github.com/chenhsiu48/PytorchWCT/raw/master/samples/street-ink4.png) 

### NTU Video
