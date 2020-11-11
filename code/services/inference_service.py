import requests
import os
import logging as log
from typing import List, Tuple
from models.image import Image

from services.image_processing_service import ImageProcessingService
from utils.exceptions import (
    PotentialObjectNotFoundException,
    CarNotFoundException,
    LicensePlateNotFoundException,
    CharactersCouldNotBeRecognizedByOCR,
)
from models.ocr_result import OcrResult
from models.detection_result import (
    DetectionResult,
    DetectionResultsDecoder,
    Box,
)


# API for object detection (it runs TF Hub's model)
predict_api_address_env = os.environ.get("PREDICT_API")
predict_api_host = (
    predict_api_address_env
    if predict_api_address_env is not None
    else "predict-api:5000"
)

# OCR server address for sending image with license plate to recognize characters
ocr_server_address_env = os.environ.get("OCR_SERVER")
ocr_server_host = (
    ocr_server_address_env if ocr_server_address_env is not None else "ocr:5000"
)

log.basicConfig(level=log.DEBUG, format="%(asctime)s - %(message)s")
log.info("Predict API address: %s", predict_api_host)
log.info("OCR server address: %s", ocr_server_host)


class InferenceService:
    """Service responsible for obtaining license plate number

    It is done by making requests to other APIs/modules (predict-api,
    ocr-server) and performing some operations like cropping an image.
    """

    _bottom_of_car_box: Box = [0.25, 0.0, 1.0, 1.0]
    _vehicle_class_entities: List[str] = ["Car", "Bus"]
    _license_plate_class_entity: str = "Vehicle registration plate"

    def send_image_to_detector(self, image: Image) -> List[DetectionResult]:
        """Sends image to predict api, which returns JSON with predictions about objects in that image.
        Before sending image, it needs to be encoded in base64.
        """

        image_base64 = image.to_base64()

        response = requests.post(
            f"http://{predict_api_host}/predict",
            json={"imgBase64": image_base64},
        )

        results = DetectionResultsDecoder().decode(response.text)

        return results

    def send_image_to_ocr(self, image: Image) -> OcrResult:
        """Sends image to ocr-server"""

        image_base64 = image.to_base64()

        response = requests.post(
            f"http://{ocr_server_host}/ocr/base64",
            json={"b64Encoded": image_base64},
        )

        print(response.status_code)
        if response.status_code != 200:
            raise CharactersCouldNotBeRecognizedByOCR

        return OcrResult.from_json(response.json())

    def find_potential_object(
        self,
        results: List[DetectionResult],
        area_threshold: int,
        score_threshold: float,
    ) -> DetectionResult:
        """Returns one result, which area and score is above given thresholds.
        If more than one result comply with conditions, one with the biggest score is returned.

        Function doesn't filter passed results by class_entity.
        So it's needed to pass filtered ones if you want find e.g. potential car.
        """
        max_score = 0
        index = -1

        for i in range(len(results)):
            area = results[i].get_percent_of_area()
            score = results[i].score

            if score > max_score and area > area_threshold and score >= score_threshold:
                max_score = score
                index = i

        if index == -1:
            message = (
                "Not found potential object for thresholds: "
                + f"area: {area_threshold}, score: {score_threshold}"
            )

            raise PotentialObjectNotFoundException(message)

        return results[index]

    def get_results_and_cropped_image_for_potential_object_of_class_entities(
        self,
        image: Image,
        class_entities: List[str],
        area_threshold: int = 20,
        score_threshold: float = 0.2,
    ) -> Tuple[List[DetectionResult], DetectionResult, Image]:
        """Returns filtered result, result with potential object and image with that potential object.

        Sends image to predict api for getting object detection result.
        Then filters that result by class_entity (e.g. "Car")
        and trying to find potential object of class_entity.
        If potential object is found, image is cropped to its box and returned.
        """

        result = self.send_image_to_detector(image)
        result_filtered_by_class_entity = list(
            filter(
                lambda r: r.class_entity in class_entities,
                result,
            )
        )

        log.debug(result_filtered_by_class_entity)

        try:
            detection_result_of_potential_object = self.find_potential_object(
                result_filtered_by_class_entity,
                area_threshold=area_threshold,
                score_threshold=score_threshold,
            )
        except PotentialObjectNotFoundException:
            raise

        cropped_image_with_potential_object = ImageProcessingService.crop_image(
            image, detection_result_of_potential_object.box
        )

        return (
            result_filtered_by_class_entity,
            detection_result_of_potential_object,
            cropped_image_with_potential_object,
        )

    def get_license_plate_number(self, image: Image) -> OcrResult:
        """Returns object with results given by OCR after some image processing.

        At first, it looks for potential car in image.
        Then crops image to slice, which contains bottom of the car.
        Then it looks for potential license plate in that slice,
        crops that slice (so now it should be slice with license plate)
        and sends that cropped image to OCR.
        """

        try:
            (
                first_result,
                potential_car,
                cropped_car,
            ) = self.get_results_and_cropped_image_for_potential_object_of_class_entities(
                image,
                self._vehicle_class_entities,
                area_threshold=0,
                score_threshold=0.0,
            )
        except PotentialObjectNotFoundException as e:
            log.error(e)
            raise CarNotFoundException()

        image_with_boxes_for_car = (
            ImageProcessingService.draw_bounding_boxes_with_class_entity(
                image, first_result, max_number_of_boxes=5, show_only_score=True
            )
        )
        image_with_boxes_for_car.save(filename_sufix="car")

        image_links = {}
        image_links["car-boxes"] = image_with_boxes_for_car.save_to_minio(
            filename_sufix="car-boxes"
        )

        image_with_bottom_of_car = ImageProcessingService.crop_image(
            cropped_car, self._bottom_of_car_box
        )
        image_with_bottom_of_car.save(filename_sufix="bottom-car")
        image_links["cropped-car"] = cropped_car.save_to_minio(
            filename_sufix="cropped-car"
        )
        image_links["bottom-of-car"] = image_with_bottom_of_car.save_to_minio(
            filename_sufix="bottom-of-car"
        )

        try:
            (
                second_result,
                potential_license_plate,
                cropped_license_plate,
            ) = self.get_results_and_cropped_image_for_potential_object_of_class_entities(
                image_with_bottom_of_car,
                [self._license_plate_class_entity],
                area_threshold=0,
                score_threshold=0.0,
            )
        except PotentialObjectNotFoundException as e:
            log.error(e)
            raise LicensePlateNotFoundException

        cropped_license_plate.save(filename_sufix="cropped-license-plate")
        image_links["cropped-license-plate"] = cropped_license_plate.save_to_minio(
            filename_sufix="cropped-license-plate"
        )

        image_with_boxes_for_plate = (
            ImageProcessingService.draw_bounding_boxes_with_class_entity(
                image_with_bottom_of_car,
                second_result,
                max_number_of_boxes=5,
                show_only_score=True,
            )
        )
        image_with_boxes_for_plate.save(filename_sufix="license-plate")
        image_links["license-plate-boxes"] = image_with_boxes_for_plate.save_to_minio(
            filename_sufix="license-plate-boxes"
        )

        try:
            ocr_result = self.send_image_to_ocr(cropped_license_plate)
        except CharactersCouldNotBeRecognizedByOCR:
            raise

        log.debug(ocr_result)

        ocr_result.image_links.update(**image_links)

        return ocr_result
