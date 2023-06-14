#!/bin/sh
# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

vai_q_tensorflow quantize --input_frozen_graph ../tf_model/yolov3-obj_tf_model.pb \
			  --input_fn yolov3_graph_input_keras_fn.calib_input \
			  --output_dir ../yolov3_quantized \
	                  --input_nodes image_input \
			  --output_nodes conv2d_58/BiasAdd,conv2d_66/BiasAdd,conv2d_74/BiasAdd \
			  --input_shapes ?,416,416,3 \
			  --gpu 0 \
			  --calib_iter 100 \
