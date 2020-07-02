# Flask with TensorFlow Hub module

Simple Flask server with endpoint for uploading a photo (from url). The photo is then loaded to module (to detect object) and saved with bounding boxes in directory results.

Tested on: 
 - Docker version: 19.03.12-ce
 - Docker Compose version: 1.26.1

## Clone

```bash
git clone https://github.com/Ar3q/flask-with-tf-hub-module.git
cd flask-with-tf-hub-module
```

## Downloading the TensorFlow Hub's model

The model can be downloaded and untared (to /tmp) with simple script:

```bash
./download-module-and-untar-to-tmp.sh
```

If something like that appears:

```bash
./download-module-and-untar-to-tmp.sh
```

You need add execution right:

```bash
chmod +x ./download-module-and-untar-to-tmp.sh
```

And try running once again


## Local development with docker-compose

Nvidia Container Toolkit doesn't work with docker-compose yet ([Github issue](https://github.com/docker/compose/issues/6691)), it's needed to use nvidia-docker [Do you support Docker Compose?](https://github.com/NVIDIA/nvidia-docker/wiki#do-you-support-docker-compose)

After installing nvidia-docker, restart a docker service and check if nvidia is shown as runtime:

```bash
sudo systemctl restart docker.service
docker info | grep Runtimes
```

Then to run container with automatic restart after every save in index.py type:

```bash
docker-compose -f docker-compose.dev.yml up
```

## Running

### Run with Docker

1. On CPU
	1. Build an image:
		```bash
		./build-docker-image-cpu.sh
		```
	2. Run a container:
        ```bash
		./run-docker-cpu.sh
		```
2. On GPU (needed installed Nvidia driver, [more information](https://www.tensorflow.org/install/docker#gpu_support))
	1. Build an image:
		```bash
		./build-docker-image-gpu.sh
		```
	2. Run a container:
        ```bash
		./run-docker-gpu.sh
		```
	3. Possible troubles:
		- `E tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR`
		If something like this appears in logs, stop container and start new one.

### Run without Docker

1. Create and activate virtual environment with virtualenv:
	```bash
	virtualenv -p python3 env
       source ./env/bin/activate
	```
2. Install dependencies:
   	```bash
       pip install tensorflow==2.0.1
       pip install -r requirements.txt
	```
3. Manually change line 150 in index.py file with correct path to model (if you used script from the first step it should be: /tmp/openimages_v4__ssd__mobilenet_v2):
   	```python
      	detector = hub.load("/tmp/openimages_v4__ssd__mobilenet_v2").signatures['default'] 
	```
4. Run a flask server:
	```bash
       FLASK_APP="index.py" flask run --host=0.0.0.0
	```

### Send request (I'm using [Insomnia](https://insomnia.rest/), but Postman or curl should also do their job ;))

1. Set Content-Type header to application/json:
       ![content type](/screens/content-type1.png)
2. Add to JSON key url with string value of url with photo:
       ![JSON body](/screens/json-body1.png)
3. Send request (to http://localhost:5000/upload)
4. If everything run correctly (processing will take some time, especially for a few first requests), you will find the photo with drawn bounding boxes in results directory:
       ![result](/screens/result-1.png)