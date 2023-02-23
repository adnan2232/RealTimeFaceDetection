from encode_faces import encode_faces_facenet
from save_load_encoding import load_encoding_json
from joblib import load
from sklearn.preprocessing import Normalizer
from time import time
import numpy as np
import csv
import cv2 as cv

class FaceRecognition:

    base_threshold = {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75}

    thresholds = {
		'VGG-Face': {'cosine': 0.40, 'euclidean': 0.60, 'euclidean_l2': 0.86},
        'Facenet':  {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 0.80},
        'Facenet512':  {'cosine': 0.30, 'euclidean': 23.56, 'euclidean_l2': 1.04},
        'ArcFace':  {'cosine': 0.68, 'euclidean': 4.15, 'euclidean_l2': 1.13},
        'Dlib': 	{'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.4},
        'SFace': 	{'cosine': 0.5932763306134152, 'euclidean': 10.734038121282206, 'euclidean_l2': 1.055836701022614},
		'OpenFace': {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
		'DeepFace': {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64},
		'DeepID': 	{'cosine': 0.015, 'euclidean': 45, 'euclidean_l2': 0.17}
    }

    def __init__(self,*arg,**kwarg):
        self.queue = kwarg["queue"]
        self.classifier = kwarg["classifier"] if "classifier" in kwarg else "cosine"

        if self.classifier == "svm":
            self.load_svm(kwarg["clf_file"],kwarg["name_enc_file"])
        elif self.classifier == "knn":
            self.knn = self.load_knn(kwarg["clf_file"],kwarg["name_enc_file"])
        else:
            self.names, self.features = self.load_encoding(kwarg["encoding_path"])
            print(self.names)
            self.total_faces = len(self.names)
        
        self.in_encoder = Normalizer(norm="l2")
        self.seen_file = open("face_seen_fn512_l2.csv","a") 
        self.writer = csv.writer(self.seen_file)
        self.start_recognition()

    def recognize_face(self,features,min,sec):
        if self.classifier =="svm":
            self.recognize_face_svm(features,min,sec)
        elif self.classifier == "l2":
            self.recognize_face_l2(features,min,sec)
        else:
            self.recognize_face_knn(features,min,sec)

    def recognize_face_svm(self,features,min,sec):
        features = self.in_encoder.transform([feature for feature in features])
        face_indices = self.clf.predict(features)
        names = self.name_encoded.inverse_transform(face_indices)
        for name in names:
            self.writer.writerow([name,min,sec])

    def recognize_face_knn(self,features,min,sec):

        faces = self.clf.predict(features)
        names = self.name_encoded.inverse_transform(faces)
            
        self.writer.writerows(zip(names,[min]*len(names),[sec]*len(names)))


    def recognize_face_l2(self,features,minu,sec):
    
        for feature in features:
            faces = {}
            for i in range(self.total_faces):
                if self.names[i] in faces:
                    faces[self.names[i]].append(self.euclidean_distance(self.features[i],feature))
                else:
                    faces[self.names[i]] = [self.euclidean_distance(self.features[i],feature)]

            name, threshold  = "unknown", FaceRecognition.thresholds["Facenet512"]["euclidean"]
   
            for face_name, l2 in faces.items():
                mean = np.min(l2)
                if mean <= threshold:
                    threshold = mean
                    name = face_name
            
            self.writer.writerow([name,minu,sec])
 


    def euclidean_distance(self,feature1, feature2):
     
        return np.linalg.norm(feature1-feature2)

    def load_encoding(self,encoding_path):
        return load_encoding_json(encoding_path)

    def load_svm(self,clf_path,name_enc_path):
        self.clf_file = clf_path
        self.name_enc_file = name_enc_path
        self.name_encoded = self.load_name_encoder(self.name_enc_file)
        self.clf = self.load_classifier(self.clf_file)
        
    def load_knn(self,clf_path,name_enc_path):
        self.name_encoded = self.load_name_encoder(name_enc_path)
        self.clf = self.load_classifier(clf_path)

    def load_classifier(self,file_name):
        return load(file_name)
    
    def load_name_encoder(elf,file_name):
        return load(file_name)

    def face_encodings(self,frame,bboxes):
        faces = [(frame[y:h,x:w],left_eye,right_eye) for x,y,w,h,left_eye,right_eye in bboxes]
        return encode_faces_facenet([face for face in faces if (len(face[0].shape)>2 and face[0].shape[0]!=0 and face[0].shape[1]!=0)])

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
                
