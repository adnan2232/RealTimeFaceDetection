from joblib import Parallel,delayed
import mediapipe as mp
import numpy as np
import cv2 as cv
from mtcnn_cv2 import MTCNN
from PyQt5.QtCore import  QThread, pyqtSignal

class VideoStream(QThread):
    stream_signal = pyqtSignal(np.ndarray)

    def __init__(self,MPqueue,*arg,**kwargs) -> None:
        self.face_detector_model = kwargs["detection_model"]
        super(VideoStream,self).__init__()
        self.IP = kwargs["IP"]
        self.username = kwargs["username"]
        self.password = kwargs["password"]
        self.MPqueue = MPqueue
        self._run_flag = True
    
            
    def get_bbox_mediapipe(self,face,img_row,img_col):
        rrb = face.location_data.relative_bounding_box
        x,y = int(img_col*rrb.xmin),int(img_row*rrb.ymin)
        width,height = int(img_col*rrb.width),int(img_row*rrb.height)
<<<<<<< HEAD
        cv.rectangle(
            frame, 
            (x,y),
            (x+width,y+height),
            color = (0,255,0),
            thickness=2
        )
    
    def draw_bbox_mtcnn(self,frame,face):
        x,y,w,h = face['box']
        cv.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
=======

        landmarks = face.location_data.relative_keypoints
        right_eye = (int(landmarks[0].x * img_col), int(landmarks[0].y * img_row))
        left_eye = (int(landmarks[1].x * img_col), int(landmarks[1].y * img_row))
        return [(x,y),(x+width,y+height),(left_eye,right_eye)]
    
    def get_bbox_mtcnn(self,face):
        x,y,width,height = face['box']
        left_eye, right_eye = face["keypoints"]["left_eye"], face["keypoints"]["right_eye"]
        return [(x,y),(x+width,y+height),(left_eye,right_eye)]
>>>>>>> 162aafd192e430a707f0d66f692dc5ac0e9c0105

    def get_bbox(self,face,img_row,img_col):
        if self.face_detector_model == "mediapipe":
            return self.get_bbox_mediapipe(face,img_row,img_col)
        else:
            return self.get_bbox_mtcnn(face)

    def FaceDetection(self):
        if self.face_detector_model == "mediapipe":
            return mp.solutions.face_detection.FaceDetection(
            model_selection = 1
            )
        else:
            return MTCNN()

    def capture_faces(self,frame):
        if self.face_detector_model == "mediapipe":
            return self.face_detector.process(frame).detections
        else:
            return self.face_detector.detect_faces(frame)

    def run(self):
        self.face_detector = self.FaceDetection()
        self.capture = cv.VideoCapture(f"rtsp://{self.username}:{self.password}@{self.IP}:554/stream1")
        fps = self.capture.get(cv.CAP_PROP_FPS)
        frame_no = 0
        faces = None
        while(True):
            isframe, frame = self.capture.read()
            if not isframe:
                continue

            frame_no += 1
            if frame_no%fps == 0:
                frame_no = 0

            img_row, img_col = frame.shape[0],frame.shape[1]
            frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

            if frame_no%5 == 0:
                faces = self.capture_faces(frame) 
            
            if not faces:
                continue
            bboxes = []
            
            for face in faces:
                (x,y),(w,h),(left_eye,right_eye) = self.get_bbox(face,img_row,img_col)
                cv.rectangle(
                    frame, 
                    (x,y),
                    (w,h),
                    color = (0,255,0),
                    thickness=2
                )
                bboxes.append([x,y,w,h,left_eye,right_eye])

            self.stream_signal.emit(frame)
            if not self.MPqueue.empty():
                self.MPqueue.get()
            


    def stop(self):
        self.capture.release()
        self._run_flag=False
        self.wait()
