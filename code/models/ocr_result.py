from dataclasses import dataclass
from typing import List


@dataclass
class OcrResult:
    number_from_separate_characters: str
    numbers_from_raw_image: List[str]

    @classmethod
    def from_json(cls, json_response: dict):
        return cls(
            json_response["numberFromSeparateChars"],
            json_response["numbersFromOCR"],
        )
