from flask import Flask, request

import tensorflow as tf

# For running inference on the TF-Hub module.
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

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

IMG_HEIGHT = 720
IMG_WIDTH = 1080


def download_and_resize_image(
    url, new_width=256, new_height=256, display=False
):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(
        pil_image, (new_width, new_height), Image.ANTIALIAS
    )
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    log.info("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename


def format_and_resize_image(data, new_width=256, new_height=256):
    image_data = BytesIO(data)
    pil_image = Image.open(image_data)

    now = time.time()
    filename = f"./upload/{now}.jpg"
    pil_image.save(filename, format="JPEG", quality=90)
    return filename


def draw_bounding_box_on_image(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    color,
    font,
    thickness=4,
    display_str_list=(),
):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (
        xmin * im_width,
        xmax * im_width,
        ymin * im_height,
        ymax * im_height,
    )
    draw.line(
        [
            (left, top),
            (left, bottom),
            (right, bottom),
            (right, top),
            (left, top),
        ],
        width=thickness,
        fill=color,
    )

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and log.info from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(
                class_names[i].decode("ascii"), int(100 * scores[i])
            )
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str],
            )
            np.copyto(image, np.array(image_pil))
    return image


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
    inference_result, for_class="", area_threshold=1
):
    max_area = 0
    index = -1

    for i in range(len(inference_result["detection_area"])):
        area = inference_result["detection_area"][i]
        if (
            for_class == ""
            or for_class == inference_result["detection_class_entities"][i]
        ):
            if area > max_area and area > area_threshold:
                max_area = area
                index = i

    if index != -1:
        return (
            inference_result["detection_boxes"][index],
            inference_result["detection_scores"][index],
        )

    return np.array([]), np.array([])


# Object detection module
start_time = time.time()
detector = hub.load("/model").signatures["default"]
end_time = time.time()
log.info("Loading module time: %.2f", end_time - start_time)


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def run_detector(detector, path, save=False):
    img = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[
        tf.newaxis, ...
    ]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    log.info("Found %d objects.", len(result["detection_scores"]))
    log.info("Inference time: %.2f", end_time - start_time)

    cars = filter_cars(result)
    log.debug(cars)

    image_with_boxes = draw_boxes(
        img.numpy(),
        cars["detection_boxes"],
        cars["detection_class_entities"],
        cars["detection_scores"],
    )

    if save:
        now_time = time.time()
        path_to_save = f"./results/{now_time}.jpg"
        log.debug("Path to save: %s", path_to_save)
        image_to_save = Image.fromarray(image_with_boxes)
        image_to_save.save(path_to_save)

    car_area, car_score = get_coordinates_and_score_of_the_biggest_area(
        cars, for_class=b"Car"
    )

    if car_area.size != 0 and car_score.size != 0:
        image_with_the_biggest_area_of_car = draw_boxes(
            img.numpy(),
            np.array([car_area]),
            np.array([b"Car"]),
            np.array([car_score]),
        )

        if save:
            path_to_save = f"./results/{now_time}-car.jpg"
            log.debug("Path to save: %s", path_to_save)
            image_to_save = Image.fromarray(image_with_the_biggest_area_of_car)
            image_to_save.save(path_to_save)


@app.route("/")
def hello():
    return "Hello, World!"


@app.route("/upload", methods=["POST"])
def photo():
    image_url = request.json["url"]
    downloaded_image_path = download_and_resize_image(
        image_url, IMG_WIDTH, IMG_HEIGHT
    )
    log.debug("downloaded path: %s", downloaded_image_path)
    run_detector(detector, downloaded_image_path, True)
    return "ok"


@app.route("/upload2", methods=["POST"])
def test2():
    uploaded_photo_path = format_and_resize_image(
        request.data, IMG_WIDTH, IMG_HEIGHT
    )
    log.debug("uploaded path: %s", uploaded_photo_path)
    run_detector(detector, uploaded_photo_path, True)
    return "ok"
