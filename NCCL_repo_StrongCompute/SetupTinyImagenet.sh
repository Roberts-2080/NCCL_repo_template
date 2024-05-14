#!/bin/bash

url='https://image-net.org/data/tiny-imagenet-200.zip'
zipname='tiny-imagenet-200.zip'
dirname='tiny-imagenet-200'

# Download tiny imagenet zip if not present
if [ ! -f $zipname ] && [ ! -d $dirname ]; then
	wget $url
fi

# Unzip
if [ ! -d $dirname ]; then
	unzip $zipname
fi

# Run restructure script on data
if [ -f $dirname/val/val_annotations.txt ]; then
	python3 RestructureValData.py
fi
