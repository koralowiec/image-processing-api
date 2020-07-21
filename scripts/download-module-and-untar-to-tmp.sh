#!/bin/bash

function download_and_untar {
    module_name=$1
    module_download_url=$2

    echo $module_name $module_download_url

    wget $module_download_url -O /tmp/$module_name.tar.gz
    mkdir /tmp/$module_name/
    tar -xf /tmp/$module_name.tar.gz -C /tmp/$module_name/
}

download_and_untar openimages_v4__ssd__mobilenet_v2 https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1?tf-hub-format=compressed

download_and_untar faster_rcnn_openimages_v4_inception_resnet_v2 https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1?tf-hub-format=compressed