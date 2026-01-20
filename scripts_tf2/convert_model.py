import os
import sys
import numpy as np
from utils import getval_cfg
from quantizer_func import quantizer
import argparse

parser = argparse.ArgumentParser(description='Darknet To xmodel Converter.')

parser.add_argument('--config_name', help='Darknet cfg file name.')
parser.add_argument('--weights_name', help='Darknet weights file name.')

args = parser.parse_args()

# default directory
config_dir = "../work"
weights_dir = "../work/backup"
out_dir = "../keras_model"

# config path of darknet
config_path = os.path.join(config_dir, args.config_name)
if not os.path.exists(config_path):
    print(f'ERROR: {config_path} does NOT exists.')
    exit(1)

# weight file of darknet
weights_path = os.path.join(weights_dir, args.weights_name)
if not os.path.exists(weights_path):
    print(f'ERROR: {weight_path} does NOT exists.')
    exit(1)

# output file name of keras model
h5_name = os.path.splitext(args.weights_name)[0] + ".h5"
h5_path = os.path.join(out_dir, h5_name)
if not os.path.exists(out_dir):
    print(f"ERROR: {out_dir} does NOT exists.")
    exit(1)
if os.path.exists(h5_path):
    os.remove(h5_path)
keras_converter = "../keras-YOLOv3-model-set/tools/model_converter/convert.py"
keras_arguments = " ".join([config_path,weights_path,h5_path])
cmd = " ".join(["python3",keras_converter,keras_arguments])

# RUN conversion
print(cmd)
os.system(cmd)

if not os.path.exists(h5_path):
    print(f'ERROR: Could not convert into keras_model [{h5_path}].\n')
else:
    print(f'#### Successfully converted darknet model into keras model {h5_path} ####\n')

# extract parameter used in quantizer
width = getval_cfg(config_path, 'width')
height = getval_cfg(config_path, 'height')

quantized_dir = "../yolov3_quantized2"
quantized_name = "quantized_" + h5_name
quantized_path = os.path.join(quantized_dir, quantized_name)
if os.path.exists(quantized_path):
    os.remove(quantized_path)

# RUN quantizer
quantizer(keras_model=h5_path, quantized_path=quantized_path, in_shape=[height,width,3])
if not os.path.exists(quantized_path):
    print(f'ERROR: Could not make quantized file [{quantized_path}].\n')
    exit(1)
else:
    print(f'#### Successfully made quantized model {quantized_path} ####\n')

# Set argument of compiler
arch = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
compiled_dir = "../compiled_yolov3"
compiled_name = os.path.splitext(args.weights_name)[0]
if not os.path.exists(compiled_dir):
    os.makedirs(compiled_dir)
last_arg = f"{{'mode':'normal','save_kernel':'', 'input_shape':'1,{height},{width},3'}}"
last_arg = '"' + last_arg + '"'
compile_args = f"-m {quantized_path} -a {arch} -o {compiled_dir} -n {compiled_name} -e {last_arg}"
cmd = f"vai_c_tensorflow2 {compile_args}"
print(cmd)
os.system(cmd)

compiled_path = os.path.join(compiled_dir, compiled_name + ".xmodel")
if not os.path.exists(compiled_path):
    print(f'ERROR: Could not compile file [{compiled_path}].\n')
    exit(1)
else:
    print(f'#### Successfully compiled quantized model {compiled_path} ####\n')

