from dataclasses import dataclass
from typing import List


@dataclass
class OcrResult:
    license_plate_number: str
    image_links: dict

    @classmethod
    def from_json(cls, json_response: dict):
        return cls(json_response["licensePlateNumber"], json_response["links"])
