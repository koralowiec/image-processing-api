from tensorflow.image import crop_to_bounding_box
from typing import List
from models.image import Image


class ImageProcessingService:
    @staticmethod
    def crop_image(image: Image, box: List[float]) -> Image:
        image_tensor = image.image_tensor
        img_height = image_tensor.shape[0]
        img_width = image_tensor.shape[1]

        ymin, xmin, ymax, xmax = tuple(box)
        offset_height = int(ymin * img_height)
        offset_width = int(xmin * img_width)
        target_height = int((ymax - ymin) * img_height)
        target_width = int((xmax - xmin) * img_width)

        cropped_image_tensor = crop_to_bounding_box(
            image_tensor,
            offset_height,
            offset_width,
            target_height,
            target_width,
        )

        return Image(cropped_image_tensor)
