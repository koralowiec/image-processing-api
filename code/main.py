from fastapi import FastAPI, Depends, File
from pydantic import BaseModel

from models.image import Image
from services.inference_service import InferenceService

app = FastAPI()


@app.get("/healthcheck")
def healthcheck():
    return "OK"


class UploadUrlReqBody(BaseModel):
    url: str


@app.post("/upload")
def photo(
    uploadUrlReqBody: UploadUrlReqBody,
    inference_service: InferenceService = Depends(),
):
    image_url = uploadUrlReqBody.url
    image = Image.from_url(image_url)
    image.save(directory="upload")

    return inference_service.get_license_plate_number(image)


@app.post("/upload2")
def raw_image(
    image: bytes = File(...), inference_service: InferenceService = Depends(),
):
    image = Image.from_raw(image)
    image.save(directory="upload")

    return inference_service.get_license_plate_number(image)
