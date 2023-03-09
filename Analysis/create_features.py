from os import listdir, mkdir,rmdir, path
from shutil import rmtree
import cv2 as cv
from mtcnn_cv2 import MTCNN
import mediapipe as mp
from encode_faces import encode_faces
from save_load_encoding import save_encoding_json
import json
import sys


class MediaPipeWrapper:

    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection = 1
        )

    def get_bbox(self,face,img_row,img_col):
        rrb = face.location_data.relative_bounding_box
        x,y = int(img_col*rrb.xmin),int(img_row*rrb.ymin)
        width,height = int(img_col*rrb.width),int(img_row*rrb.height)

        landmarks = face.location_data.relative_keypoints
        right_eye = (int(landmarks[0].x * img_col), int(landmarks[0].y * img_row))
        left_eye = (int(landmarks[1].x * img_col), int(landmarks[1].y * img_row))
        return [(x,y),(x+width,y+height),(left_eye,right_eye)]


    def capture_faces(self,frame):
        return self.face_detector.process(frame).detections



class MTCNNWrapper:

    def __init__(self):
        self.face_detector = MTCNN()
        
    def get_bbox(self,face,img_row,img_col):
        x,y,width,height = face['box']
        left_eye, right_eye = face["keypoints"]["left_eye"], face["keypoints"]["right_eye"]
        return [(x,y),(x+width,y+height),(left_eye,right_eye)]


    def capture_faces(self,frame):
        return self.face_detector.detect_faces(frame)

def create_features(file_path):
    mediapipe = MediaPipeWrapper()
    faces_ls,names = [],[] 
    for folder in listdir(file_path):
        name = folder.lower()
        cropped_folder = file_path+"/"+"cropped_folder"+"_"+name
        '''try:
            rmtree(cropped_folder)
            rmdir(cropped_folder)
        except:
            pass
        mkdir(cropped_folder)'''
        for file in listdir(file_path+"/"+folder):
            img = cv.imread(file_path+"/"+folder+"/"+file)
            img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            faces = mediapipe.capture_faces(img)
            if not faces:
                continue
            for face in faces:
                (x,y),(w,h),(left_eye,right_eye) = mediapipe.get_bbox(face,img.shape[0],img.shape[1])
                
                cr_face = img[y:h,x:w]
                if cr_face.shape[0]!=0 and cr_face.shape[1]!=0:
                    faces_ls.append((cr_face,left_eye,right_eye))
                    names.append(name)
                    #cv.imwrite(cropped_folder+"/"+file,cr_face)

    features_ls = encode_faces(faces_ls)

    save_encoding_json(features_ls,names,"feature_encoding.json")

if __name__ == "__main__":
    file_path = "Faces/lfw_home/tmoc_faces"
    create_features(file_path)