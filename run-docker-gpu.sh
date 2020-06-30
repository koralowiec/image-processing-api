#!/bin/bash

current_path=$(pwd)
module_name=openimages_v4__ssd__mobilenet_v2
model_path=/tmp/$module_name/

docker run --gpus all \
--rm --init -it \
-p 5000:5000 \
--mount type=bind,\
source=$model_path,\
target=/model \
--mount type=bind,\
source=$current_path/results,\
target=/src/results \
--mount type=bind,\
source=$current_path/upload,\
target=/src/upload \
tf-gpu-flask