#!/bin/bash 

current_path=$(pwd) 
module_name_ssd=openimages_v4__ssd__mobilenet_v2
model_path_ssd=/tmp/$module_name_ssd/

module_name_faster_rcnn=openimages_v4__ssd__mobilenet_v2
model_path_faster_rcnn=/tmp/$module_name_faster_rcnn/

docker run --gpus all \
--rm --init -it \
-p 5000:5000 \
-e MODULE=true \
--mount type=bind,\
source=$model_path_ssd,\
target=/model_ssd \
--mount type=bind,\
source=$model_path_faster_rcnn,\
target=/model_faster_rcnn \
--mount type=bind,\
source=$current_path/../results,\
target=/src/results \
--mount type=bind,\
source=$current_path/../upload,\
target=/src/upload \
tf-gpu-flask:base