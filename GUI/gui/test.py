from FaceDetector.fd_mtcnn import MTCNNWrapper
from FaceRecognizer.fr_template import FaceRecogTemp
import cv2 as cv

fd =  MTCNNWrapper()
fr = FaceRecogTemp(model_name="Dlib")

print([enc["name"] for enc in fr.encodings])

fr.delete_encoding("jethalal")

print([enc["name"] for enc in fr.encodings])