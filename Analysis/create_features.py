from os import listdir
import cv2 as cv
from mtcnn_cv2 import MTCNN
from encode_faces import encode_faces_facenet
from save_load_encoding import load_encoding_json, save_encoding_json
import json
import sys

def create_features(file_path):
    mtcnn = MTCNN()

    faces_ls,names = [],[] 
    for file in listdir(file_path):

        img = cv.imread(file_path+"/"+file)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        faces = mtcnn.detect_faces(img)
        name = file.split(".")[0]
        for face in faces:
            x,y,w,h = face["box"]
            cr_face = img[x:x+w,y:y+h]
            if cr_face.shape[0]!=0 and cr_face.shape[1]!=0:
                faces_ls.append(cr_face)
                names.append(name)

    features_ls = encode_faces_facenet(faces_ls)
    save_encoding_json(features_ls,names,"feature_encoding.json")

if __name__ == "__main__":
    file_path = sys.argv[1]
    create_features(file_path)