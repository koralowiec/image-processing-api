from dataclasses import dataclass


@dataclass
class OcrResult:
    """Class for ocr-server's API response

    Args:
        license_plate_number (str): Recognized license plate number
        image_links (dict): Dictionary with links to images uploaded to minio
            by ocr-server
    """

    license_plate_number: str
    image_links: dict

    @classmethod
    def from_json(cls, json_response: dict):
        return cls(json_response["licensePlateNumber"], json_response["links"])
