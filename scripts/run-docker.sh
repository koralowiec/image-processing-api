#!/bin/bash

current_path=$(pwd)
docker run \
--rm --init -it \
-p 5000:5000 \
--mount type=bind,\
source=$current_path/../results,\
target=/src/results \
--mount type=bind,\
source=$current_path/../upload,\
target=/src/upload \
tf-cpu-flask