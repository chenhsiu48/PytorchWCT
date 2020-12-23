#!/bin/bash

for x in input/*.jpg; do 
    for e in water oil pencil ink; do
        ./WCT.py --content $x --effect $e
    done
done
