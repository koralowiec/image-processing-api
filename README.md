# Image processing API

It's purpose is to process received image: find a car in the picture, then find a license plate and return number of that plate. 
It uses other API/modules for object detection ([predict-api](https://github.com/koralowiec/predict-api)) and OCR ([ocr-server](https://github.com/koralowiec/ocr-server))

## Clone

```bash
git clone https://github.com/koralowiec/image-processing-api
cd image-processing-api
```

## Local development with docker-compose

```bash
docker-compose -f ./docker/docker-compose.dev.yml up
```
