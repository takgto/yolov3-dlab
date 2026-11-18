import os, sys
import numpy as np
import keras
import tensorflow

#from config import cifar10_config as cfg
CALIB_DATASET = '/workspace/yolov3-dlab/scripts_tf2/my_tf_calib4.npy'

def quantizer(keras_model='../keras_model/yolov3-obj.h5', quantized_path='../yolov3_quantized2/quantized_yolov3-obj.h5', 
        in_shape=[416,416,3], cal_steps=10, cal_batch_size=8):
    quantized_dir = os.path.dirname(quantized_path)
    model = keras.models.load_model(keras_model)
    # Inspector
    from tensorflow_model_optimization.quantization.keras import vitis_inspect
    #inspector = vitis_inspect.VitisInspector(target="DPUCADF8H_ISA0")
    target_kv260 = '/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json'
    inspector = vitis_inspect.VitisInspector(target=target_kv260) # not worked by KV260
    inspector.inspect_model(model,
                            input_shape=in_shape,
                            #input_shape=[416,416,3],
                            plot=True,
                            plot_file=os.path.join(quantized_dir, "model.svg"),
                            dump_results=True,
                            dump_results_file=os.path.join(quantized_dir, "inspect_results.txt"),
                            verbose=0)

    # Quantizer
    # 注意: 既存の npy (my_tf_calib4.npy) は BGR / /256 で作られている。
    # KV260 の C++ (setInputPointer) は RGB / INTER_LINEAR / /256 なので、
    # 正しい前処理の npy を使うこと。
    # 正しい npy の作成: yolo-quant-eval/scripts/make_calib_npy.py を使う。
    print('loading CALIB_DATASET...')
    valid_dataset = np.load(CALIB_DATASET)
    # 既存 npy が BGR の場合は RGB に変換する
    # valid_dataset = valid_dataset[:, :, :, ::-1]  # BGR→RGB（チャンネル軸を反転）
    print('done.')

    #cal_steps = args['cal_steps']
    #cal_batch_size = args['cal_batch_size']

    if len(valid_dataset)==0:
        print('validataion data set 0, cannnot apply quantizer!')
    else:
        from tensorflow_model_optimization.quantization.keras import vitis_quantize
        quantizer = vitis_quantize.VitisQuantizer(model)
        quantized_model = quantizer.quantize_model(calib_dataset=valid_dataset,
                                               calib_steps=cal_steps,
                                               calib_batch_size=cal_batch_size,
                                               verbose=0)
        quantized_model.save(quantized_path)

