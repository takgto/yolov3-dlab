# yolov3-dlab
Many codes related to yolov3 and its quantization tool, edge application and some useful scripts. 

# Quantization Scripts

## tensorflow1 base
  **convert_yolov3.sh** in scripts folder convert darknet file(*.weights) into keras file(*.h5) with ../keras-YOLOv3-model-set/tools. The keras file is saved on ../karas_model directory. Its h5 file is also converted into tensorflow file(*.pb) in ../tf_model directory.
  
  **qunatize_yolov3.sh** in scripts folder quantizes the tensorflow file(*.pb). vai_q_tensorflow in this script needs yolov3_graph_input_keras_fn python file and this function requires validation files under ./val2017 folder and yolov3_tf_calib file. dbthe list of validation files. The resulting quantized file *.pb (float) is stored on ../yolov3_quantized directory.
  
  **compile_yolov3** in scripts compiles quantized *.pb file, resulting in another quantized file of type int8 under ../yolov3_compiled directory.
  
## tensorflow2 base
  **my_quantizer.py** in scripts_tf2 folder quantizes tensorflow file(*.pb) based on vai_q_tensorflow2.
  The command:
    python my_quantizer.py -i
    -i means execute inspector which shows each layer computes in DPU or CPU.
    my_quantizer needs numpy file which is done by some preprocession. Prepares my_tf_calib.npy and my_tf_calib2.npy. *2 is half data of *1.
    
  **compile_yolov3.sh** in scripts_tf2 folder compiles quantized *.pb file, resulting in another quantized file of type int8 under ../yolov3_compiled directory.
  
## work
  There are some files in darknet.
