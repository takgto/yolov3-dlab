import os, sys
import numpy as np
import keras
import tensorflow

#from config import cifar10_config as cfg
YOLOV3_DIR = '/workspace/yolov3-dlab/' 
KERAS_MODEL = 'yolov3-obj.h5'
CALIB_DATASET = '/workspace/yolov3-dlab/scripts_tf2/my_tf_calib2.npy'
board={}
board['U280']=board['U50']=board['U50LV']='DPUCAHX8L'
board['VCK190']='DPUCVDX8G'
board['KV260']=board['ZCU102']=board['ZCU104']='DPUCZDX8G'

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i",  "--inspect", action='store_true', default=False, help="use inspect model or not")
ap.add_argument("-n",  "--network", default="yolov3",             help="input CNN")
ap.add_argument("-s",  "--cal_steps", type=int, default=10,     help="# of epochs")
ap.add_argument("-bs", "--cal_batch_size", type=int, default=16, help="size of mini-batches passed to network")
args = vars(ap.parse_args())

inspect = args["inspect"]
network = args["network"]
quantized_dir = os.path.join(YOLOV3_DIR, 'yolov3_quantized2')

# load kera model trained by cifar10 dataset.
mdel_name = os.path.join(YOLOV3_DIR, 'keras_model', KERAS_MODEL)
print(model_name)
model = keras.models.load_model(model_name)

# Inspector
if inspect:
    from tensorflow_model_optimization.quantization.keras import vitis_inspect
    #inspector = vitis_inspect.VitisInspector(target="DPUCADF8H_ISA0")
    target_kv260 = '/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json'
    inspector = vitis_inspect.VitisInspector(target=target_kv260) # not worked by KV260
    inspector.inspect_model(model,
                            input_shape=[416,416,3],
                            plot=True,
                            plot_file=os.path.join(quantized_dir, "model.svg"),
                            dump_results=True,
                            dump_results_file=os.path.join(quantized_dir, "inspect_results.txt"),
                            verbose=0)

# Quantizer
print('loading CALIB_DATASET...')
valid_dataset = np.load(CALIB_DATASET)
print('done.')

cal_steps = args['cal_steps']
cal_batch_size = args['cal_batch_size']

if len(valid_dataset)==0:
    print('validataion data set 0, cannnot apply quantizer!')
else:
    from tensorflow_model_optimization.quantization.keras import vitis_quantize
    quantizer = vitis_quantize.VitisQuantizer(model)
    quantized_model = quantizer.quantize_model(calib_dataset=valid_dataset,
                                           calib_steps=cal_steps,
                                           calib_batch_size=cal_batch_size,
                                           verbose=0)
    quantized_model.save(os.path.join(quantized_dir, 'quantized_model.h5'))

