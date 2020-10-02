from fastapi import FastAPI, Depends, File
from pydantic import BaseModel

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio

# For downloading the image.
from six.moves.urllib.request import urlopen

import numpy as np

# For measuring the inference time.
import time

import itertools

import os
import logging as log

import requests
import base64
import json

debug = os.environ.get("DEBUG")
log.basicConfig(
    level=log.DEBUG if debug else log.INFO, format="%(asctime)s - %(message)s"
)

app = FastAPI()

# Print Tensorflow version
log.info("TensorFlow version: %s", tf.__version__)

# OCR server address for sending image with license plate to recognize characters
ocr_server_address_env = os.environ.get("OCR_SERVER")
ocr_server_address = (
    ocr_server_address_env if ocr_server_address_env is not None else "ocr:5000"
)
log.info("OCR server address: %s", ocr_server_address)

# API for object detection (it runs TF Hub's model)
predict_api_address_env = os.environ.get("PREDICT_API")
predict_api_address = (
    predict_api_address_env if predict_api_address_env is not None else ""
)
log.info("Predict API address: %s", predict_api_address)


def download_image_from_url_and_save(url):
    response = urlopen(url)
    image_data = response.read()
    directory = "upload"
    filepath = save_img(image_data, raw=True, directory=directory)
    log.info("Image downloaded to %s.", filepath)
    return filepath


def draw_boxes_with_text(
    image, boxes, class_names, scores, max_boxes=10, min_score=0.1,
):
    img = image
    class_names_iterator = iter(class_names)
    scores_iterator = iter(scores)
    text = ""
    encoding = "utf-8"
    colors = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 0.1, 0.5],
            [1.0, 0.5, 1.0],
            [0.5, 0.5, 0.5],
            [0.2, 0.7, 0.5],
            [0.5, 0.1, 0.1],
            [0.2, 0.2, 1.0],
        ]
    )
    color_pool = itertools.cycle(colors)

    box_i = 1

    for box in boxes:
        score = next(scores_iterator)
        if score >= min_score and box_i <= max_boxes:
            class_name = next(class_names_iterator)
            score = "{:.0f}".format(score * 100)
            text = f"{class_name} {score}%"

            box = np.array([[box]])

            color = np.array([next(color_pool)])
            img_4D = tfa.image.utils.to_4D_image(img)
            img_4D = tf.image.convert_image_dtype(img_4D, tf.float32)

            img_b = tfio.experimental.image.draw_bounding_boxes(
                img_4D, box, colors=color, texts=[text]
            )
            img_b = tf.image.convert_image_dtype(img_b, tf.uint8)
            img_b = tfa.image.utils.from_4D_image(img_b, 3)

            box_i = box_i + 1
            img = img_b

    return img


def draw_boxes(
    image, boxes, class_names, scores,
):
    boxes_np = np.reshape(boxes, (-1, boxes.shape[0], boxes.shape[1]))

    colors = np.array(
        [[1.0, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.1, 0.0], [0.0, 0.5, 1.0]]
    )
    img_4D = tfa.image.utils.to_4D_image(image)
    img_4D = tf.image.convert_image_dtype(img_4D, tf.float32)
    log.debug(img_4D)

    img_b = tf.image.draw_bounding_boxes(img_4D, boxes_np, colors)
    img_b = tf.image.convert_image_dtype(img_b, tf.uint8)
    img_b = tfa.image.utils.from_4D_image(img_b, 3)

    return img_b


def compute_area(coordinates):
    ymin, xmin, ymax, xmax = tuple(coordinates)
    height = ymax - ymin
    width = xmax - xmin
    return height * width


def filter_by_detection_class_entities(
    inference_result, entities, min_score=0.0
):
    detection_class_entities = np.array([], dtype=object)
    detection_class_names = np.array([], dtype=object)
    detection_boxes = np.array([[0, 0, 0, 0]])
    detection_scores = np.array([], dtype="float32")
    detection_class_labels = np.array([])
    detection_area = np.array([])

    for i in range(len(inference_result["detection_class_entities"])):
        if inference_result["detection_scores"][i] >= min_score:
            if inference_result["detection_class_entities"][i] in entities:
                area_percent = (
                    compute_area(inference_result["detection_boxes"][i]) * 100
                )

                detection_class_entities = np.append(
                    detection_class_entities,
                    [inference_result["detection_class_entities"][i]],
                )
                detection_class_names = np.append(
                    detection_class_names,
                    [inference_result["detection_class_names"][i]],
                )
                detection_boxes = np.vstack(
                    [detection_boxes, inference_result["detection_boxes"][i]]
                )
                detection_scores = np.append(
                    detection_scores, [inference_result["detection_scores"][i]]
                )
                detection_class_labels = np.append(
                    detection_class_labels,
                    [inference_result["detection_class_labels"][i]],
                )
                detection_area = np.append(detection_area, [area_percent])

    detection_boxes = np.delete(detection_boxes, (0), axis=0)

    return {
        "detection_class_entities": detection_class_entities,
        "detection_class_names": detection_class_names,
        "detection_boxes": detection_boxes,
        "detection_scores": detection_scores,
        "detection_class_labels": detection_class_labels,
        "detection_area": detection_area,
    }


def filter_cars(inference_result, min_score=0.1):
    return filter_by_detection_class_entities(inference_result, ["Car"])


def filter_license_plates(inference_result, min_score=0.1):
    return filter_by_detection_class_entities(
        inference_result, ["Vehicle registration plate"]
    )


def get_coordinates_and_score_of_the_biggest_area(
    inference_result, area_threshold, score_threshold, for_class=""
):
    max_area = 0
    index = -1

    for i in range(len(inference_result["detection_area"])):
        if (
            for_class == ""
            or inference_result["detection_class_entities"][i] == for_class
        ):
            area = inference_result["detection_area"][i]
            score = inference_result["detection_scores"][i]

            if (
                area > max_area
                and area > area_threshold
                and score >= score_threshold
            ):
                max_area = area
                index = i

    if index != -1:
        return (
            inference_result["detection_boxes"][index],
            inference_result["detection_scores"][index],
        )

    return np.array([]), np.array([])


def load_img_from_fs(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def save_img(
    img, raw=False, filename="", directory="results", filename_sufix=""
):
    if img is None:
        raise ValueError("There is no image to save", img)

    if raw:
        img = tf.io.decode_jpeg(img, channels=3)

    img_jpeg = tf.io.encode_jpeg(img, quality=100)

    if not filename:
        now = time.time()
        filename = f"{now}{filename_sufix}"

    filepath = f"./{directory}/{filename}.jpeg"
    tf.io.write_file(filepath, img_jpeg)
    return filepath


def send_image_to_detector(img, raw=False):
    if not raw:
        img = tf.io.encode_jpeg(img, quality=100)
        img = img.numpy()

    img_b64 = base64.b64encode(img)
    img_dec = img_b64.decode("utf-8")

    r = requests.post(
        f"http://{predict_api_address}/predict", json={"imgBase64": img_dec},
    )

    return r.json()


def get_results_from_detector(img):
    start_time = time.time()
    result = send_image_to_detector(img)
    end_time = time.time()

    log.info("Found %d objects.", len(result["detection_scores"]))
    log.info("Inference time: %.2f", end_time - start_time)
    log.debug(result)

    return result


def crop_img(img, box):
    img_height = img.shape[0]
    img_width = img.shape[1]

    ymin, xmin, ymax, xmax = tuple(box)
    offset_height = int(ymin * img_height)
    offset_width = int(xmin * img_width)
    target_height = int((ymax - ymin) * img_height)
    target_width = int((xmax - xmin) * img_width)

    log.debug("Cropped image")
    log.debug("Top left (x,y): %d, %d", offset_width, offset_height)
    log.debug("Height: %d, width: %d", target_height, target_width)

    return tf.image.crop_to_bounding_box(
        img, offset_height, offset_width, target_height, target_width
    )


def draw_boxes_with_objects_for_class(
    inference_result, img, for_class, min_score=0.1, max_boxes=20,
):
    potential_objects = filter_by_detection_class_entities(
        inference_result, [for_class]
    )
    log.debug(potential_objects)

    image_with_boxes = draw_boxes_with_text(
        img,
        potential_objects["detection_boxes"],
        potential_objects["detection_class_entities"],
        potential_objects["detection_scores"],
        min_score=min_score,
        max_boxes=max_boxes,
    )

    img_path = save_img(
        image_with_boxes, filename_sufix=f"boxes_for_{for_class}"
    )
    log.debug("Path to img: %s", img_path)


def crop_or_draw_box_with_potential_detected_object(
    inference_result, img, for_class, area_threshold=20, score_threshold=0.4
):
    potential_objects = filter_by_detection_class_entities(
        inference_result, [for_class]
    )

    object_area, object_score = get_coordinates_and_score_of_the_biggest_area(
        potential_objects,
        for_class=for_class,
        area_threshold=area_threshold,
        score_threshold=score_threshold,
    )

    if object_area.size != 0 and object_score.size != 0:
        image_with_potential_object_in_bounding_box = draw_boxes_with_text(
            img,
            np.array([object_area]),
            np.array([for_class]),
            np.array([object_score]),
        )

        cropped_image_with_potential_object = crop_img(img, object_area)

        return (
            cropped_image_with_potential_object,
            image_with_potential_object_in_bounding_box,
        )

    log.info("Potential object not found")
    return None, None


def get_cropped_and_drawn_images_for_potential_detected_object(
    inference_result,
    image,
    for_class,
    save=False,
    area_threshold=20,
    score_threshold=0.4,
):
    cropped, drawn = crop_or_draw_box_with_potential_detected_object(
        inference_result,
        image,
        for_class,
        area_threshold=area_threshold,
        score_threshold=score_threshold,
    )

    if save:
        try:
            save_img(drawn, filename_sufix=f"boxes_for_{for_class}")
            save_img(cropped, filename_sufix=f"cropped_for_{for_class}")
        except ValueError as e:
            log.error(e)

    return cropped, drawn


def inference_image(image):
    results = get_results_from_detector(image)

    draw_boxes_with_objects_for_class(results, image, "Car")
    (
        cropped,
        drawn,
    ) = get_cropped_and_drawn_images_for_potential_detected_object(
        results, image, "Car"
    )

    bottom_of_cropped_car = crop_img(cropped, [0.25, 0.0, 1.0, 1.0])
    save_img(bottom_of_cropped_car, filename_sufix="bottom")

    # 2nd inference
    results = get_results_from_detector(bottom_of_cropped_car)
    log.debug(results)
    cropped, drawn = get_cropped_and_drawn_images_for_potential_detected_object(
        results,
        bottom_of_cropped_car,
        "Vehicle registration plate",
        True,
        score_threshold=0.05,
        area_threshold=0,
    )

    return cropped


def inference_image_with_cropping(image):
    results = get_results_from_detector(image)
    draw_boxes_with_objects_for_class(results, image, "Car")
    (
        cropped,
        drawn,
    ) = get_cropped_and_drawn_images_for_potential_detected_object(
        results, image, "Car"
    )

    bottom_of_cropped_car = crop_img(cropped, [0.25, 0.0, 1.0, 1.0])
    save_img(bottom_of_cropped_car, filename_sufix="bottom")

    pieces = crop_to_pieces(bottom_of_cropped_car)

    for piece in pieces:
        results = get_results_from_detector(piece)
        (
            cropped,
            drawn,
        ) = get_cropped_and_drawn_images_for_potential_detected_object(
            results,
            piece,
            "Vehicle registration plate",
            True,
            score_threshold=0.05,
            area_threshold=0,
        )


def send_image_to_ocr(img, raw=False):
    if not raw:
        img = tf.io.encode_jpeg(img, quality=100)
        img = img.numpy()

    img_b64 = base64.b64encode(img)
    img_dec = img_b64.decode("utf-8")

    r = requests.post(
        f"http://{ocr_server_address}/ocr/base64", json={"b64Encoded": img_dec}
    )
    return r.json()


def crop_to_pieces(img):
    pieces = []

    pieces.append(crop_and_save(img, [0.0, 0.0, 0.5, 0.5]))
    pieces.append(crop_and_save(img, [0.0, 0.5, 0.5, 1.0]))
    pieces.append(crop_and_save(img, [0.5, 0.0, 1.0, 0.5]))
    pieces.append(crop_and_save(img, [0.5, 0.5, 1.0, 1.0]))

    pieces.append(crop_and_save(img, [0.0, 0.25, 0.5, 0.75]))
    pieces.append(crop_and_save(img, [0.25, 0.25, 0.75, 0.75]))
    pieces.append(crop_and_save(img, [0.5, 0.25, 1.0, 0.75]))
    return pieces


def crop_and_save(img, box):
    cropped = crop_img(img, box)
    save_img(cropped, directory="results/cropped", filename_sufix="cropped")
    return cropped


@app.get("/")
def hello():
    return "Hello, World!"


class UploadUrlReqBody(BaseModel):
    url: str


@app.post("/upload")
def photo(uploadUrlReqBody: UploadUrlReqBody):
    image_url = uploadUrlReqBody.url
    downloaded_image_path = download_image_from_url_and_save(image_url)
    log.debug("downloaded path: %s", downloaded_image_path)
    image = load_img_from_fs(downloaded_image_path)

    image_with_plate = inference_image(image)

    numbers: dict = {}
    if image_with_plate is not None:
        numbers = send_image_to_ocr(image_with_plate)

    return numbers


@app.post("/upload2")
def raw_image(image: bytes = File(...)):
    uploaded_photo_path = save_img(image, raw=True, directory="upload")
    log.debug("uploaded path: %s", uploaded_photo_path)
    image = tf.image.decode_jpeg(image, channels=3)

    image_with_plate = inference_image(image)

    numbers: dict = {}
    if image_with_plate is not None:
        numbers = send_image_to_ocr(image_with_plate)

    return numbers


@app.post("/crop")
def cropping(uploadUrlReqBody: UploadUrlReqBody):
    image_url = uploadUrlReqBody.url
    downloaded_image_path = download_image_from_url_and_save(image_url)
    log.debug("downloaded path: %s", downloaded_image_path)
    image = load_img_from_fs(downloaded_image_path)

    inference_image_with_cropping(image)

    return "ok"
