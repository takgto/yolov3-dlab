#!/bin/bash
./convert_yolov3.sh
python my_quantizer.py -i
./compile_yolov3.sh
cd ../compiled
ls

