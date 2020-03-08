# Flask with TensorFlow Hub module

Simple Flask server with endpoint for uploading a photo (from url). The photo is then loaded to module (to detect object) and saved with bounding boxes in directory results.

## Clone

```shell
git clone https://github.com/Ar3q/flask-with-tf-hub-module.git
cd flask-with-tf-hub-module
```

## Running

1. Download module to disk (to directory: /tmp)

```shell
./download-module-and-untar-to-tmp.sh
```

If something like that appears:

```shell
bash: ./download-module-and-untar-to-tmp.sh: Permission denied
```

You need add execution right:

```shell
chmod +x ./download-module-and-untar-to-tmp.sh
```

And try running once again

2. Run with Docker
	1. On CPU
		1. Build an image:
			```shell
			./build-docker-image-cpu.sh
			```
		2. Run a container:
            ```shell
			./run-docker-cpu.sh
			```
     1. On GPU (needed installed Nvidia driver, [more information](https://www.tensorflow.org/install/docker#gpu_support))
		1. Build an image:
			```shell
			./build-docker-image-gpu.sh
			```
		2. Run a container:
            ```shell
			./run-docker-gpu.sh
			```
3. Run without Docker
	1. Create and activate virtual environment with virtualenv:
		```shell
		virtualenv -p python3 env
        source ./env/bin/activate
		```
    2. Install dependencies:
    	```shell
        pip install tensorflow==2.0.1
        pip install -r requirements.txt
		```
    3. Manually change line 150 in index.py file with correct path to model (if you used script from the first step it should be: /tmp/openimages_v4__ssd__mobilenet_v2):
    	```python
       	detector = hub.load("/tmp/openimages_v4__ssd__mobilenet_v2").signatures['default'] 
		```
	4. Run a flask server:
		```shell
        FLASK_APP="index.py" flask run --host=0.0.0.0
		```
4. Send request (I'm using [Insomnia](https://insomnia.rest/), but Postman or curl should also do their job ;))
	1. Set Content-Type header to application/json:
        ![content type](/screens/content-type1.png)
	2. Add to JSON key url with string value of url with photo:
        ![JSON body](/screens/json-body1.png)
	3. Send request
	4. If everything run correctly (processing will take some time, especially for a few first requests), you will find the photo with drawn bounding boxes in results directory:
        ![result](/screens/result-1.png)