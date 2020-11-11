from dataclasses import dataclass
from six.moves.urllib.request import urlopen
from tensorflow import Tensor, float32, uint8
from tensorflow.image import decode_jpeg, encode_jpeg, convert_image_dtype
from tensorflow.io import read_file, write_file
from tensorflow_addons.image.utils import to_4D_image, from_4D_image
import base64
import time
import numpy as np
import uuid
import io
from minio.error import ResponseError

from utils.minio_setup import minio_client, bucket_name


@dataclass
class Image:
    """Class for image used in processing"""

    image_tensor: Tensor

    @classmethod
    def from_raw(cls, image_bytes: bytes):
        image_tensor = decode_jpeg(image_bytes, channels=3)
        return cls(image_tensor)

    @classmethod
    def from_url(cls, url: str):
        response = urlopen(url)
        image_data = response.read()
        return cls.from_raw(image_data)

    @classmethod
    def from_base64(cls, image_encoded: str):
        image_decoded = base64.b64decode(image_encoded)
        return cls.from_raw(image_decoded)

    @classmethod
    def from_file_system(cls, path: str):
        image = read_file(path)
        return cls.from_raw(image)

    @classmethod
    def from_4D_float32_tensor(cls, image_4D_f32: Tensor):
        image_unint8 = convert_image_dtype(image_4D_f32, uint8)
        image = from_4D_image(image_unint8, 3)
        return cls(image)

    def save(
        self,
        filename: str = "",
        directory: str = "results",
        filename_sufix: str = "",
    ) -> str:
        """Saves image to file system"""

        image_jpeg = encode_jpeg(self.image_tensor, quality=100)

        if not filename:
            now = time.time()
            filename = f"{now}{filename_sufix}"

        filepath = f"./{directory}/{filename}.jpeg"
        write_file(filepath, image_jpeg)

        return filepath

    def save_to_minio(self, filename_sufix: str = "") -> str:
        """Saves image to minio (object storage service)"""
        image_bytes = self.to_bytes()

        uid = uuid.uuid1()
        if filename_sufix != "":
            filename_sufix = f"-{filename_sufix}"
        filename = f"{uid}{filename_sufix}.jpeg"
        path = f"{bucket_name}/{filename}"

        try:
            with io.BytesIO(image_bytes) as data:
                _ = minio_client.put_object(
                    bucket_name, filename, data, len(image_bytes)
                )
        except ResponseError as e:
            print(e)

        return path

    def to_numpy_array(self) -> np.ndarray:
        img_jpeg = encode_jpeg(self.image_tensor, quality=100)
        return img_jpeg.numpy()

    def to_base64(self) -> str:
        img_np = self.to_numpy_array()
        img_b64 = base64.b64encode(img_np)
        return img_b64.decode("utf-8")

    def to_4D_float32(self) -> Tensor:
        img_4D = to_4D_image(self.image_tensor)
        img_4D_f32 = convert_image_dtype(img_4D, float32)
        return img_4D_f32

    def to_bytes(self) -> bytes:
        image_jpeg = encode_jpeg(self.image_tensor, quality=100)
        image_bytes = image_jpeg.numpy()
        return image_bytes

    def to_raw_str(self) -> str:
        image_bytes = self.to_bytes()
        return str(image_bytes)
