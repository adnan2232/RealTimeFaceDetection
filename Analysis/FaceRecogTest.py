from face_recognition.api import face_encodings,face_distance
from joblib import Parallel, delayed
from time import time
import numpy as np
import csv
import json

class FaceRecognition:

    def __init__(self,*arg,**kwarg):
        self.queue = kwarg["queue"]

        self.faces_file = kwarg["file_name"]
        self.face_name, self.db_features = self.load_db_faces()
        self.start_recognition()

    def recognize_face(self,feature,min,sec):
        face_distances = face_distance(self.db_features,feature)
        face_index = np.argmin(face_distances)
    
        if face_distances[face_index]<=0.6:
            with open("face_seen.csv","a") as seen:
                writer = csv.writer(seen)
                writer.writerow([self.face_name[face_index],min,sec,face_distances[face_index]])

    def load_db_faces(self):
        name, features = [],[]

        with open(self.faces_file,"r") as fc:
            faces_info= json.load(fc)

        for face_info in faces_info:
            name.append(face_info["name"])
            features.append(face_info["features"])

        return name,np.array(features)

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

                face_features = face_encodings(frame,bboxes,3)
                Parallel(n_jobs=-1,prefer="threads")(delayed(self.recognize_face)(feature,min,sec) for feature in face_features)

        except KeyboardInterrupt:
            print("recognition stop")

        finally:
            print(f"Recognizer total time: {(time()-start_time_recog)//60}, Seconds: {(time()-start_time_recog)%60}\n")
                
