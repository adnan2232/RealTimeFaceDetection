from deepface.detectors.FaceDetector import alignment_procedure
from sklearn.neighbors import KNeighborsClassifier
from .distance_metrics import match
from typing import Union
from tinydb import TinyDB, Query
from deepface import DeepFace
import numpy as np
from joblib import Parallel, delayed


class FaceRecogTemp:

    models = ["Facenet", "Facenet512", "ArcFace"]
    thresholds_global = {
        "Facenet": {"cosine": 0.40, "l2": 10, "l2_norm": 0.70},
        "Facenet512": {"cosine": 0.30, "l2": 23.56, "l2_norm": 0.80},
        "ArcFace": {"cosine": 0.68, "l2": 4.15, "l2_norm": 1.0},
        
    }
    

    def __init__(self, model_name: str = "Facenet") -> None:

        if model_name not in FaceRecogTemp.models:
            raise ValueError(f"No model name: {model_name}")

        self.model_name = model_name
        self.knn = KNeighborsClassifier(n_neighbors=3,weights="distance")
        self.encodings = self.fetch_encoding()
        self.thresholds = FaceRecogTemp.thresholds_global[model_name]
        self.fit_knn()

  
    def fetch_encoding(self) -> list[dict]:
        
        return TinyDB(f"{self.model_name}.db").all()

    def create_encoding(
        self,
        img: np.array,
        left_eye: np.int8,
        right_eye: np.int8,
    ) -> Union[list, np.array]:

        try:
            al_img = alignment_procedure(img, left_eye, right_eye)
        except Exception:
            al_img = img
        finally:
            try:
                
                enc = DeepFace.represent(
                    img_path=self.normalize_input(al_img),
                    model_name=self.model_name,
                    detector_backend="skip",
                )
             
                return np.array(enc[0]["embedding"]) / np.linalg.norm(enc[0]["embedding"])
            
            except:
                
                return np.array([])

    def create_encodings(self, img: np.array, bboxes: list[list[int]]) -> list[list]:

        res = Parallel(n_jobs=-1,prefer="threads")(
            delayed(self.create_encoding)(
                img[bbox["y"] : bbox["h"], bbox["x"] : bbox["w"]],
                bbox["left_eye"],
                bbox["right_eye"],
            )
            for bbox in bboxes
        )

        return res

    def save_encoding(self, name: str, encoding: Union[list, np.array]) -> None:
        TinyDB(f"{self.model_name}.db").insert({"name": name, "encoding": encoding})
    

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

        encoding = self.create_encoding(img[y:h, x:w], left_eye, right_eye).tolist()
       
        if len(encoding)==0:
            return
        self.save_encoding(name, encoding)

    def delete_encoding(self, name: str) -> None:
       
        TinyDB(f"{self.model_name}.db").remove(Query().name == name)
    
        

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
  
    def fit_knn(self) -> None:
        if len(self.encodings) > 0:
            encs,names =[],[]
            for enc in self.encodings:
                if len(enc["encoding"]) >0:
                    encs.append(enc["encoding"])
                    names.append(enc["name"])
            if names!=[]:
                self.knn.fit(
                    encs,
                    names,
                )

    def predicts(self, vec: list[np.array]) -> list[list[str, float]]:
        if len(self.encodings) <= 0 or len(vec) <= 0:
            return []
        res_name = []
        res_conf = []
        dist = self.knn.kneighbors(vec)[0]
        name = self.knn.predict(vec)
        n = len(vec)
     
        for i in range(n):

            if round(np.min(dist[i]), 2) <= self.thresholds["l2_norm"]:
               
                res_name.append(name[i])
                res_conf.append(round(np.min(dist[i]), 2))

        return [res_name, res_conf]

    def normalize_input(self, img):

        if self.model_name == "base" or self.model_name == "Dlib":
            return img

        elif self.model_name == "Facenet":
            mean, std = img.mean(), img.std()
            img = (img - mean) / std

        elif self.model_name == "Facenet512":

            img = img/127.5
            img = img-1

        elif self.model_name == "ArcFace":

            img = img-127.5
            img = img/128
        else:
            raise ValueError(f"unimplemented normalization type - {self.model_name}")

        return img
