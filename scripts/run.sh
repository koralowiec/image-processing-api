#!/bin/bash

chmod +x *.sh

d_flag=false
b_flag=false
g_flag=false

while getopts 'bdg' flag; do
	case "${flag}" in
		b) b_flag=true ;;
		d) d_flag=true ;;
		g) g_flag=true ;;
		*) exit 1 ;;
	esac
done

if [ "$d_flag" = true ] ; then
	./download-module-and-untar-to-tmp.sh
fi

if [ "$b_flag" = true ] ; then
	if [ "$g_flag" = true ] ; then
		./build-docker-image-gpu.sh
	else
		./build-docker-image-cpu.sh
	fi
fi

if [ "$g_flag" = true ] ; then
	./run-docker-gpu.sh
else
	./run-docker-cpu.sh
fi