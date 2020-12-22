from typing import List
from dataclasses import dataclass
from json import JSONDecoder
import logging as log


@dataclass
class NumberCheckResult:
    id: int
    number: str


class NumberCheckResultDecoder(JSONDecoder):
    def __init__(self):
        JSONDecoder.__init__(self, object_hook=self.dict_to_list)

    def dict_to_list(self, json_response: dict) -> List[NumberCheckResult]:
        numbers = []
        if not isinstance(json_response, list):
            json_response = [json_response]

        for n in json_response:
            n_id = n["id"]
            n_number = n["number"]

            number = NumberCheckResult(n_id, n_number)
            numbers.append(number)

        return numbers
