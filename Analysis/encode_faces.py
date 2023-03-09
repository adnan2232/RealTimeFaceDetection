import numpy as np
import cv2 as cv
import os
import traceback
from deepface import DeepFace
from deepface.detectors.FaceDetector import alignment_procedure

def encode_faces(faces,model_name="ArcFace"):

    encodings = []
    for i in range(len(faces)):
    
        face, left_eye,right_eye = faces[i][0],faces[i][1], faces[i][2]
        aligned_face = alignment_procedure(face,left_eye,right_eye)

        emb = DeepFace.represent(
                aligned_face,
                model_name=model_name,
                detector_backend="skip",
                enforce_detection=False,
            )[0]["embedding"]

        encodings.append(
            (emb/np.linalg.norm(emb)).tolist()
        )

    return encodings