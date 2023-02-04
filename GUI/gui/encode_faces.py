import numpy as np
import cv2 as cv
import traceback
from deepface import DeepFace
from deepface.detectors.FaceDetector import alignment_procedure

def encode_faces(faces,model_name):

    encodings = []
    for i in range(len(faces)):
        try:
            face, left_eye,right_eye = faces[i][0],faces[i][1], faces[i][2]
            encodings.append(
                DeepFace.represent(
                    alignment_procedure(face,left_eye,right_eye),
                    model_name="model_name",
                    detector_backend="skip",
                    enforce_detection=False
                )
            )
        except:
            pass

    return encodings