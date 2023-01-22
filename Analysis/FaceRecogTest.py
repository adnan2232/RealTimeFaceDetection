from encode_faces import encode_faces_facenet
from joblib import load
from sklearn.preprocessing import Normalizer
from time import time
import numpy as np
import csv
import json

class FaceRecognition:

    def __init__(self,*arg,**kwarg):
        self.queue = kwarg["queue"]

        self.clf_file = kwarg["clf_file"]
        self.name_enc_file = kwarg["name_enc_file"]

        self.name_encoded = self.load_name_encoder(self.name_enc_file)
        self.clf = self.load_classifier(self.clf_file)
        self.in_encoder = Normalizer(norm="l2")
        self.seen_file = open("face_seen.csv","a") 
        self.writer = csv.writer(self.seen_file)
        self.start_recognition()

    def recognize_face(self,features,min,sec):
        features = self.in_encoder.transform([feature for feature in features])
        face_indices = self.clf.predict(features)
        names = self.name_encoded.inverse_transform(face_indices)
        for name in names:
            self.writer.writerow([name,min,sec])

    def load_classifier(self,file_name):
        return load(file_name)
    
    def load_name_encoder(elf,file_name):
        return load(file_name)

    def load_db_faces(self):
        name, features = [],[]

        with open(self.faces_file,"r") as fc:
            faces_info= json.load(fc)

        for face_info in faces_info:
            name.append(face_info["name"])
            features.append(face_info["features"])

        return name,np.array(features)

    def face_encodings(self,frame,bboxes):
        faces = [frame[x:x+w,y:y+h] for x,y,w,h in bboxes]
        return encode_faces_facenet([face for face in faces if (face.shape[0]!=0 and face.shape[1]!=0)])

    def start_recognition(self):
        try:
            start_time_recog = time()
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
            print(f"Recognizer total time: {(time()-start_time_recog)//60}, Seconds: {(time()-start_time_recog)%60}\n")
                
