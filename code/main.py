from fastapi import FastAPI, Depends, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models.image import Image
from services.inference_service import InferenceService
from utils.exceptions import (
    CarNotFoundException,
    LicensePlateNotFoundException,
    CharactersCouldNotBeRecognizedByOCR,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/docs/pdoc/", StaticFiles(directory="docs"))


@app.get("/healthcheck")
def healthcheck():
    return "OK"


class UploadUrlReqBody(BaseModel):
    url: str


class Base64Body(BaseModel):
    b64Encoded: str = Field(..., title="Image encoded in Base64")


def get_lp_number(image: Image, inference_service: InferenceService) -> dict:
    """Tries to get license plate number for given image

    Returns:
        str: license plate number
    """
    try:
        dict_with_results = inference_service.get_license_plate_number(image)
    except CarNotFoundException:
        raise HTTPException(
            status_code=422,
            detail="Could not find a potential car in the sent image",
        )
    except LicensePlateNotFoundException:
        raise HTTPException(
            status_code=422,
            detail="Could not find a license plate in the sent image",
        )
    except CharactersCouldNotBeRecognizedByOCR:
        raise HTTPException(
            status_code=422,
            detail="Could not recognize characters in the sent image",
        )

    return dict_with_results


@app.post("/upload/url")
def url_image(
    uploadUrlReqBody: UploadUrlReqBody,
    inference_service: InferenceService = Depends(),
):
    image_url = uploadUrlReqBody.url
    image = Image.from_url(image_url)
    image.save(directory="upload")

    return get_lp_number(image, inference_service)


@app.post("/upload/raw")
async def raw_image(
    request: Request,
    inference_service: InferenceService = Depends(),
):
    image = await request.body()
    image = Image.from_raw(image)
    image.save(directory="upload")

    return get_lp_number(image, inference_service)


@app.post("/upload/base64")
def base64_image(
    base64ReqBody: Base64Body,
    inference_service: InferenceService = Depends(),
):
    image = Image.from_base64(base64ReqBody.b64Encoded)
    image.save(directory="upload")

    return get_lp_number(image, inference_service)
