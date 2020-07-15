from flask import Flask, request

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio
import tensorflow_hub as hub

# For downloading the image.
from six.moves.urllib.request import urlopen

import numpy as np

# For measuring the inference time.
import time

import itertools

import os
import logging as log

debug = os.environ.get("DEBUG")
log.basicConfig(
    level=log.DEBUG if debug else log.INFO, format="%(asctime)s - %(message)s"
)

app = Flask(__name__)

# https://github.com/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb

# Print Tensorflow version
log.info("TensorFlow version: %s", tf.__version__)

# Check available GPU devices.
log.info(
    "The following GPU devices are available: %s" % tf.test.gpu_device_name()
)


# Object detection module
start_time = time.time()
detector = hub.load("/model").signatures["default"]
end_time = time.time()
log.info("Loading module time: %.2f", end_time - start_time)


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
            class_name = str(next(class_names_iterator), encoding)
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
    inference_result, entities, min_score=0.1
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
    return filter_by_detection_class_entities(inference_result, [b"Car"])


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


def save_img(img, raw=False, filename="", directory="results"):
    if raw:
        img = tf.io.decode_jpeg(img, channels=3)

    img_jpeg = tf.io.encode_jpeg(img)

    if not filename:
        now = time.time()
        filename = f"{now}"

    filepath = f"./{directory}/{filename}.jpeg"
    tf.io.write_file(filepath, img_jpeg)
    return filepath


def run_detector(detector, img):
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[
        tf.newaxis, ...
    ]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

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


def draw_boxes_with_cars(inference_result, img, save=False):
    cars = filter_cars(inference_result)
    log.debug(cars)

    image_with_boxes = draw_boxes_with_text(
        img,
        cars["detection_boxes"],
        cars["detection_class_entities"],
        cars["detection_scores"],
        min_score=0.1,
        max_boxes=20,
    )

    if save:
        img_path = save_img(image_with_boxes)
        log.debug("Path to img: %s", img_path)


def crop_or_draw_box_with_potential_car(
    inference_result, img, area_threshold=20, score_threshold=0.4
):
    cars = filter_cars(inference_result)
    car_area, car_score = get_coordinates_and_score_of_the_biggest_area(
        cars,
        for_class=b"Car",
        area_threshold=area_threshold,
        score_threshold=score_threshold,
    )

    if car_area.size != 0 and car_score.size != 0:
        image_with_potential_car_in_bounding_box = draw_boxes(
            img,
            np.array([car_area]),
            np.array([b"Car"]),
            np.array([car_score]),
        )

        cropped_image_with_potential_car = crop_img(img, car_area)

        log.debug("CROPPED")
        log.debug(cropped_image_with_potential_car)

        return (
            cropped_image_with_potential_car,
            image_with_potential_car_in_bounding_box,
        )

    log.info("Potential car not found")
    return None, None


@app.route("/")
def hello():
    return "Hello, World!"


@app.route("/upload", methods=["POST"])
def photo():
    image_url = request.json["url"]
    downloaded_image_path = download_image_from_url_and_save(image_url)
    image = load_img_from_fs(downloaded_image_path)
    log.debug("downloaded path: %s", downloaded_image_path)
    results = run_detector(detector, image)
    draw_boxes_with_cars(results, image, True)
    cropped, drawn = crop_or_draw_box_with_potential_car(results, image, True)
    save_img(cropped)
    save_img(drawn)
    return "ok"


@app.route("/upload2", methods=["POST"])
def raw_image():
    uploaded_photo_path = save_img(request.data, raw=True, directory="upload")
    img = tf.image.decode_jpeg(request.data, channels=3)
    log.debug(img)
    log.debug("uploaded path: %s", uploaded_photo_path)
    results = run_detector(detector, img)
    draw_boxes_with_cars(results, img, True)
    cropped, _ = crop_or_draw_box_with_potential_car(results, img, True)
    save_img(cropped)
    return "ok"
