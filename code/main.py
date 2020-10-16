from fastapi import FastAPI, Depends, File, HTTPException
from pydantic import BaseModel

from models.image import Image
from services.inference_service import InferenceService
from utils.exceptions import (
    CarNotFoundException,
    LicensePlateNotFoundException,
)

app = FastAPI()


@app.get("/healthcheck")
def healthcheck():
    return "OK"


class UploadUrlReqBody(BaseModel):
    url: str


def get_lp_number(image: Image, inference_service: InferenceService) -> str:
    try:
        lp = inference_service.get_license_plate_number(image)
    except CarNotFoundException:
        raise HTTPException(
            status_code=422,
            detail="Could not find a potential car in the send image",
        )
    except LicensePlateNotFoundException:
        raise HTTPException(
            status_code=422,
            detail="Could not find a license plate in the send image",
        )

    return lp


@app.post("/upload")
def photo(
    uploadUrlReqBody: UploadUrlReqBody,
    inference_service: InferenceService = Depends(),
):
    image_url = uploadUrlReqBody.url
    image = Image.from_url(image_url)
    image.save(directory="upload")

    return get_lp_number(image, inference_service)


@app.post("/upload2")
def raw_image(
    image: bytes = File(...), inference_service: InferenceService = Depends(),
):
    image = Image.from_raw(image)
    image.save(directory="upload")

    return get_lp_number(image, inference_service)
