from PyQt5.QtCore import  QThread, pyqtSignal
from encode_faces import encode_faces
from save_load_encoding import load_encoding_json
from joblib import load
from sklearn.preprocessing import Normalizer
from time import time
import numpy as np
import csv


class FaceRecognition(QThread):

    def __init__(self,*arg,**kwarg):

        super(FaceRecognition,self).__init__()
        self.queue = kwarg["MPqueue"]
        self.knn= None
        self.seen_file = open("face_seen.csv","a") 
        self.writer = csv.writer(self.seen_file)


    def load_knn(self,encoding_path):
        return load_encoding_json(encoding_path)

    def recognize_face(self,features,minu,sec):
    
        for feature in features:
            pass
            self.writer.writerow([name,minu,sec,threshold])

    def load_classifier(self,file_name):
        return load(file_name)

    def face_encodings(self,frame,bboxes):
        faces = [(frame[x:x+w,y:y+h],left_eye,right_eye) for x,y,w,h,left_eye,right_eye in bboxes]
        return encode_faces_facenet(faces)

    def run(self):
        try:
            
            while(True):
                
                if self.queue.empty():
                    continue
                    
                frame,bboxes,con, (min,sec) = self.queue.get()

                if not con:
                    
                    self.queue.close()
                    break

                faces_features = self.face_encodings(frame,bboxes)
                if faces_features:
                    self.recognize_face(faces_features,min,sec)

        except KeyboardInterrupt:
            print("recognition stop")

        finally:
            while not self.queue.empty():
                frame,bboxes,con, (min,sec) = self.queue.get()
                faces_features = self.face_encodings(frame,bboxes)
                if faces_features:
                    self.recognize_face(faces_features,min,sec)
            self.seen_file.flush()
            self.seen_file.close()
                
