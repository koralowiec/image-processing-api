from typing import List
from dataclasses import dataclass


@dataclass
class DetectionResult:
    class_entity: str
    box: List[float]
    score: float

    def get_percent_of_area(self) -> int:
        ymin, xmin, ymax, xmax = tuple(self.box)
        height = ymax - ymin
        width = xmax - xmin
        return height * width * 100
