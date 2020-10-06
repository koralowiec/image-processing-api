from fastapi import FastAPI, Depends, File
from pydantic import BaseModel
from models.image import Image
from models.detection_result_list import DetectionResultList
import base64
import requests
import os

predict_api_address_env = os.environ.get("PREDICT_API")
predict_api_address = (
    predict_api_address_env
    if predict_api_address_env is not None
    else "predict-api:5000"
)

app = FastAPI()


@app.get("/healthcheck")
def healthcheck():
    return "OK"


class UploadUrlReqBody(BaseModel):
    url: str


def send_image_to_detector(img):
    img_b64 = base64.b64encode(img)
    img_dec = img_b64.decode("utf-8")

    r = requests.post(
        f"http://{predict_api_address}/predict", json={"imgBase64": img_dec},
    )

    return r.json()


@app.post("/upload")
def photo(uploadUrlReqBody: UploadUrlReqBody):
    image_url = uploadUrlReqBody.url
    image = Image.from_url(image_url)
    image.save(directory="upload")
    j = send_image_to_detector(image.to_numpy_array())
    r = DetectionResultList.from_json(j)
    print(r.results[0])
    return image_url


@app.post("/upload2")
def raw_image(image: bytes = File(...)):
    image = Image.from_raw(image)
    return "OK"
