from deepface.detectors.FaceDetector import alignment_procedure
from distance_metrics import match
from typing import Union
from tinydb import TinyDB, Query
from deepface import DeepFace
import numpy as np


class FacenetWrapper:

    __db = TinyDB("Facenet.db")
    __query = Query()
    __encodings = __db.all()
    __thresholds = {
        "cosine":0.40,
        "l2":10,
        "l2_norm":0.80
    }
    def __init__(self) -> None:
        pass
    
    @property
    def thresholds(self):
        return FacenetWrapper.__thresholds
    
    @classmethod
    def __fetch_encoding(cls) -> None:
        FacenetWrapper.__encodings = FacenetWrapper.__db.all()

    @property
    def encodings(self):
        return FacenetWrapper.__encodings

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
            normalization="Facenet",
        )[0]["embedding"]
    


    def create_encodings(
        self,
        img: np.array,
        bboxes:list[list[int]]
    ) -> list[np.array]:

        res = []
        for bbox in bboxes:
            res.append(
                self.create_encoding(
                    img,
                    bbox["x"],bbox["y"],
                    bbox["w"],bbox["h"],
                    bbox["left_eye"], bbox["right_eye"]
                )
            )
        
        return res

    def save_encoding(self, name: str, encoding: Union[list, np.array]) -> None:

        FacenetWrapper.__db.insert({"name": name, "encoding": encoding})

        FacenetWrapper.__encodings.append({"name": name, "encoding": encoding})

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

        encoding = self.create_encoding(img,x,y,w,h,left_eye,right_eye)
        self.save_encoding(name, encoding)


    def delete_encoding(self, name: str) -> None:

        FacenetWrapper.__db.remove(FacenetWrapper.__query.name == name)
        FacenetWrapper.__fetch_encoding()

    def verify(self,enc1: np.array,enc2: np.array,metrics:str="cosine") -> bool:

        return match(enc1,enc2,self.thresholds,metrics)