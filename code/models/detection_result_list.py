from typing import List

from models.detection_result import DetectionResult


class DetectionResultList:
    results: List[DetectionResult]

    def __init__(self, results: List[DetectionResult]):
        self.results = results

    @classmethod
    def from_json(cls, json_response: dict):
        results = []
        number_of_results = len(json_response["detection_class_entities"])
        for i in range(number_of_results):
            class_entity = json_response["detection_class_entities"][i]
            box = json_response["detection_boxes"][i]
            score = json_response["detection_scores"][i]

            detectionResult = DetectionResult(class_entity, box, score)
            results.append(detectionResult)

        return cls(results)
