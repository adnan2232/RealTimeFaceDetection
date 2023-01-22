import numpy as np
import cv2 as cv
from deepface import DeepFace


def encode_faces_facenet(faces):

    encodings = []
    for i in range(len(faces)):
        try:
            faces[i] = faces[i].astype(np.float32)
            print(faces[i].shape)
            encodings.append(
                DeepFace.represent(
                    faces[i],
                    model_name="Facenet",
                    enforce_detection=False
                )
            )
        except:
            pass

    return encodings