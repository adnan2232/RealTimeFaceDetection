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
    
    

            
    def draw_bbox_mediapipe(self,frame,face,img_row,img_col):
        rrb = face.location_data.relative_bounding_box
        x,y = int(img_col*rrb.xmin),int(img_row*rrb.ymin)
        width,height = int(img_col*rrb.width),int(img_row*rrb.height)
        cv.rectangle(
            frame, 
            (x,y),
            (x+width,y+height),
            color = (255,255,0),
            thickness=1
        )
    
    def draw_bbox_mtcnn(self,frame,face):
        x,y,w,h = face['box']
        cv.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=1)

    def draw_bbox(self,frame,face,img_row,img_col):
        if self.face_detector_model == "mediapipe":
            self.draw_bbox_mediapipe(frame,face,img_row,img_col)
        else:
            self.draw_bbox_mtcnn(frame,face)

    def FaceDetection(self):
        if self.face_detector_model == "mediapipe":
            return mp.solutions.face_detection.FaceDetection(
            model_selection = 1
            )
        else:
            return MTCNN()

    def capture_faces(self,Frame):
        if self.face_detector_model == "mediapipe":
            return self.face_detector.process(Frame).detections
        else:
            return self.face_detector.detect_faces(Frame)

    def run(self):
        self.face_detector = self.FaceDetection()

        self.capture = cv.VideoCapture(f"rtsp://{self.username}:{self.password}@{self.IP}:554/stream1")
        # self.capture = cv.VideoCapture(0)
        while(True):
            isFrame, Frame = self.capture.read()
            if not isFrame:
                continue
            img_row, img_col = Frame.shape[0],Frame.shape[1]
            Frame = cv.cvtColor(Frame,cv.COLOR_BGR2RGB)
            faces = self.capture_faces(Frame)

            if faces:
                Parallel(n_jobs=-1,prefer="threads")(delayed(self.draw_bbox)(Frame,face,img_row,img_col) for face in faces)

            self.stream_signal.emit(Frame)
            if not self.MPqueue.empty():
                self.MPqueue.get()


    def stop(self):
        self.capture.release()
        self._run_flag=False
        self.wait()
