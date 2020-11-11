from typing import List, Tuple
from dataclasses import dataclass
from json import JSONDecoder

Box = Tuple[float, float, float, float]


@dataclass
class DetectionResult:
    """Class for predict-api's API response

    Args:
        class_entity (str): Recognized class of object
        box (Box): Corg
    """

    class_entity: str
    box: Box
    score: float

    def get_percent_of_area(self) -> int:
        ymin, xmin, ymax, xmax = tuple(self.box)
        height = ymax - ymin
        width = xmax - xmin
        return height * width * 100


class DetectionResultsDecoder(JSONDecoder):
    """Class for decoding JSON given by API"""

    def __init__(self):
        JSONDecoder.__init__(self, object_hook=self.dict_to_list)

    def dict_to_list(self, json_response: dict) -> List[DetectionResult]:
        results = []
        number_of_results = len(json_response["detection_class_entities"])
        for i in range(number_of_results):
            class_entity = json_response["detection_class_entities"][i]
            box = json_response["detection_boxes"][i]
            score = json_response["detection_scores"][i]

            detectionResult = DetectionResult(class_entity, box, score)
            results.append(detectionResult)

        return results
