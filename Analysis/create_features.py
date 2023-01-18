from os import listdir
import cv2 as cv
from face_recognition.api import face_encodings
import json
import sys

def create_features(file_path):
    features_to_dump = []
    for file in listdir(file_path):

        img = cv.imread(file_path+"/"+file)
        features = face_encodings(img,num_jitters=6)[0]
        features_to_dump.append({"name":file.split(".")[0],"features":features.tolist()})

    with open("faces_name_feature.json","w") as fc:
        json.dump(features_to_dump,fc)

if __name__ == "__main__":
    file_path = sys.argv[1]
    create_features(file_path)