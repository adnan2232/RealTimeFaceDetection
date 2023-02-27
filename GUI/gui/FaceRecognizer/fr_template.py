from deepface.detectors.FaceDetector import alignment_procedure
from sklearn.neighbors import KNeighborsClassifier
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
    __knn = KNeighborsClassifier(n_neighbors=2)

    def __init__(self, model_name: str = "Facenet") -> None:

        if model_name not in FaceRecogTemp.models:
            raise ValueError(f"No model name: {model_name}")

        self.model_name = model_name

        FaceRecogTemp.__db = TinyDB(f"{model_name}.db")
        FaceRecogTemp.__encodings = FaceRecogTemp.__fetch_encoding()
        FaceRecogTemp.__thresholds = FaceRecogTemp.thresholds_global[model_name]
        FaceRecogTemp.__query = Query()
        FaceRecogTemp.__fit_knn()

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
        
        try:
            al_img = alignment_procedure(img[y:h, x:w], left_eye, right_eye)
        except Exception:
            al_img = img
        finally:
            enc = DeepFace.represent(
                img_path=self.normalize_input(al_img),
                model_name=self.model_name,
                detector_backend="skip",
            )[0]["embedding"]

            return np.array(enc)/np.linalg.norm(enc)

    def create_encodings(
        self, img: np.array, bboxes: list[list[int]]
    ) -> list[list]:
        
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
                ).tolist()
            )

        return res

    def save_encoding(self, name: str, encoding: Union[list, np.array]) -> None:

        FaceRecogTemp.__db.insert({"name": name, "encoding": encoding})

        FaceRecogTemp.__encodings.append({"name": name, "encoding": encoding})

        FaceRecogTemp.__fit_knn()

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
     
        encoding = self.create_encoding(img, x, y, w, h, left_eye, right_eye).tolist()
        self.save_encoding(name, encoding)

    def delete_encoding(self, name: str) -> None:

        FaceRecogTemp.__db.remove(FaceRecogTemp.__query.name == name)
        FaceRecogTemp.__encodings = FaceRecogTemp.__fetch_encoding()
        FaceRecogTemp.__fit_knn()

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

    @property
    def knn(self):
        return FaceRecogTemp.__knn
    
    @classmethod
    def __fit_knn(cls) -> None:
        if len(FaceRecogTemp.__encodings)>0:
            FaceRecogTemp.__knn.fit(
                [enc["encoding"] for enc in FaceRecogTemp.__encodings],[enc["name"] for enc in FaceRecogTemp.__encodings]
            )

    def predicts(self,vec: list[np.array]) -> list[list[str,float]]:
        if len(FaceRecogTemp.__encodings)<=0 or len(vec)<=0:
            return []
        res_name =[]
        res_conf = []
        dist = self.knn.kneighbors(vec)[0]
        name = self.knn.predict(vec)
        n = len(vec)
      
        for i in range(n):
    
            if round(np.min(dist[i]),2)<=self.thresholds["l2_norm"]:
                res_name.append(name[i])
                res_conf.append(round(np.min(dist[i]),2))

        return [res_name,res_conf]

    def normalize_input(self,img):

        if self.model_name == "base" or self.model_name=="Dlib":
            return img

        elif self.model_name == "Facenet":
            mean, std = img.mean(), img.std()
            img = (img - mean) / std

        elif self.model_name == "Facenet512":
            
            img /= 127.5
            img -= 1

        elif self.model_name == "ArcFace":
            
            img -= 127.5
            img /= 128
        else:
            raise ValueError(f"unimplemented normalization type - {self.model_name}")

        return img