#!/bin/bash
IMGDIR=raw/*

for file in $IMGDIR;
do
    python3 opencv-segment.py $file
done

