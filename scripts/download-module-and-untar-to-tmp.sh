#!/bin/bash

module_name=openimages_v4__ssd__mobilenet_v2
module_download_url=https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1?tf-hub-format=compressed

wget $module_download_url -O /tmp/$module_name.tar.gz
mkdir /tmp/$module_name/
tar -xf /tmp/$module_name.tar.gz -C /tmp/$module_name/
