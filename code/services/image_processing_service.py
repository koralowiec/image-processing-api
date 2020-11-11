import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import itertools
from typing import List
from models.image import Image
from models.detection_result import DetectionResult, Box


class ImageProcessingService:
    """Class with methods for processing an image

    Args:
        colors (np.ndarray): Array with colors for bounding boxes
    """

    colors: np.ndarray = np.array(
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

    @staticmethod
    def crop_image(image: Image, box: Box) -> Image:
        """Crops given image using coordinates from box

        Args:
            image (Image): Image to be cropped
            box (Box): List of coordinates:
                [top_left_corner_y, top_left_corner_x,
                bottom_right_corner_y, bottom_right_corner_x]
        """

        image_tensor = image.image_tensor
        img_height = image_tensor.shape[0]
        img_width = image_tensor.shape[1]

        ymin, xmin, ymax, xmax = tuple(box)
        offset_height = int(ymin * img_height)
        offset_width = int(xmin * img_width)
        target_height = int((ymax - ymin) * img_height)
        target_width = int((xmax - xmin) * img_width)

        cropped_image_tensor = tf.image.crop_to_bounding_box(
            image_tensor,
            offset_height,
            offset_width,
            target_height,
            target_width,
        )

        return Image(cropped_image_tensor)

    @staticmethod
    def draw_bounding_boxes(
        image: Image,
        result: List[DetectionResult],
        max_number_of_boxes: int = 10,
    ) -> Image:
        """Draws bounding boxes on given image

        Args:
            image: Given image, on which will be drawn bounding boxes
            result: List of results of object detection
            max_number_of_boxes (int): Maximum number of boxes that will be
                drawn

        Returns:
            Image: Image with drawn bounding boxes
        """
        boxes = list(map(lambda detection_result: detection_result.box, result))
        boxes = boxes[:max_number_of_boxes]

        boxes_np = np.array(boxes)
        boxes_np = np.reshape(boxes_np, (-1, boxes_np.shape[0], boxes_np.shape[1]))

        image_with_boxes_4D = tf.image.draw_bounding_boxes(
            image.to_4D_float32(), boxes_np, ImageProcessingService.colors
        )

        return Image.from_4D_float32_tensor(image_with_boxes_4D)

    @staticmethod
    def draw_bounding_boxes_with_class_entity(
        image: Image,
        result: List[DetectionResult],
        max_number_of_boxes: int = 10,
        show_only_score: bool = False,
    ) -> Image:
        """Draws bounding boxes with class and score of detected object on given image

        Args:
            image: Given image, on which will be drawn bounding boxes
            result: List of results of object detection
            max_number_of_boxes (int): Maximum number of boxes that will be
                drawn
            show_only_score (bool): If true, class of the object will not be
                drawn, only object's score

        Returns:
            Image: Image with drawn bounding boxes
        """

        color_pool = itertools.cycle(ImageProcessingService.colors)

        box_number = 0
        img = image.to_4D_float32()
        for obj in result:
            if box_number < max_number_of_boxes:
                box = obj.box
                box = np.array([[box]])
                color = np.array([next(color_pool)])
                score = "{:.0f}".format(obj.score * 100)
                text = score if show_only_score else f"{obj.class_entity} {score}"

                img = tfio.experimental.image.draw_bounding_boxes(
                    img, box, colors=color, texts=[text]
                )

                box_number += 1

        return Image.from_4D_float32_tensor(img)
