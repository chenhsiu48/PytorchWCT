#!/bin/bash

for x in input/*.jpg; do 
    ./WCT.py --content $x --style style/water.jpg
    ./WCT.py --content $x --style style/oil2.jpg
    ./WCT.py --content $x --style style/pencil.jpg --gray
    ./WCT.py --content $x --style style/ink.jpg --gray
done
