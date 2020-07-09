#!/bin/bash

docker build -t tf-gpu-flask:base -f ../docker/gpu.Dockerfile  --target base ../