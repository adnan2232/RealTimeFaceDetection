from deepface.detectors.FaceDetector import alignment_procedure
from .distance_metrics import match
from typing import Union
from tinydb import TinyDB, Query
from deepface import DeepFace
import numpy as np


class FaceRecogTemp:

    models = {"Facenet", "Facenet512", "ArcFace", "Dlib"}
    thresholds_global = {
        "Facenet": {"cosine": 0.40, "l2": 10, "l2_norm": 0.80},
        "Facenet512": {"cosine": 0.30, "l2": 23.56, "l2_norm": 1.04},
        "ArcFace": {"cosine": 0.68, "l2": 4.15, "l2_norm": 1.13},
        "Dlib": {"cosine": 0.07, "l2": 0.6, "l2_norm": 0.4},
    }

    def __init__(self, model_name: str = "Facenet") -> None:

        if model_name not in FaceRecogTemp.models:
            raise ValueError(f"No model name: {model_name}")

        self.model_name = model_name

        FaceRecogTemp.__db = TinyDB(f"{model_name}.db")
        FaceRecogTemp.__encodings = FaceRecogTemp.__fetch_encoding()
        FaceRecogTemp.__thresholds = FaceRecogTemp.thresholds_global[model_name]
        FaceRecogTemp.__query = Query()

    @classmethod
    def __fetch_encoding(cls) -> list[dict]:
        return FaceRecogTemp.__db.all()


    @property
    def thresholds(self):
        return FaceRecogTemp.__thresholds

    @property
    def encodings(self):
        return FaceRecogTemp.__encodings

    def create_encoding(
        self,
        img: np.array,
        x: np.int8,
        y: np.int8,
        w: np.int8,
        h: np.int8,
        left_eye: np.int8,
        right_eye: np.int8,
    ) -> Union[list, np.array]:

        return DeepFace.represent(
            img_path=alignment_procedure(img[y:h, x:w], left_eye, right_eye),
            model_name="Facenet",
            detector_backend="skip",
            normalization=self.model_name,
        )[0]["embedding"]

    def create_encodings(
        self, img: np.array, bboxes: list[list[int]]
    ) -> list[np.array]:

        res = []
        for bbox in bboxes:
            res.append(
                self.create_encoding(
                    img,
                    bbox["x"],
                    bbox["y"],
                    bbox["w"],
                    bbox["h"],
                    bbox["left_eye"],
                    bbox["right_eye"],
                )
            )

        return res

    def save_encoding(self, name: str, encoding: Union[list, np.array]) -> None:

        FaceRecogTemp.__db.insert({"name": name, "encoding": encoding})

        FaceRecogTemp.__encodings.append({"name": name, "encoding": encoding})

    def create_save_encoding(
        self,
        name: str,
        img: np.array,
        x: np.int8,
        y: np.int8,
        w: np.int8,
        h: np.int8,
        left_eye: np.int8,
        right_eye: np.int8,
    ) -> None:

        encoding = self.create_encoding(img, x, y, w, h, left_eye, right_eye)
        self.save_encoding(name, encoding)

    def delete_encoding(self, name: str) -> None:
        
        FaceRecogTemp.__db.remove(FaceRecogTemp.__query.name == name)
        FaceRecogTemp.__encodings = FaceRecogTemp.__fetch_encoding()

    def verify(self, enc1: np.array, enc2: np.array, metrics: str = "default") -> bool:

        if metrics == "default":
            if self.model_name == "Facenet":
                metrics = "cosine"

            elif self.model_name == "Facenet512":
                metrics = "l2_norm"

            elif self.model_name == "ArcFace":
                metrics = "cosine"

            else:
                metrics = "l2_norm"

        return match(enc1, enc2, self.thresholds, metrics)
