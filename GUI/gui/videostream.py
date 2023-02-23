from FaceDetector.fd_mediaipipe import MediaPipeWrapper
from FaceDetector.fd_mtcnn import MTCNNWrapper
import numpy as np
import cv2 as cv
from PyQt5.QtCore import  QThread, pyqtSignal

class VideoStream(QThread):
    stream_signal = pyqtSignal(np.ndarray)

    def __init__(self,*arg,**kwargs) -> None:
        self.face_detector_model = kwargs["detection_model"]
        super(VideoStream,self).__init__()
        self.IP = kwargs["IP"]
        self.username = kwargs["username"]
        self.password = kwargs["password"]
        self.MPqueue = kwargs["MPQueue"]
        self._run_flag = True
    

    def FaceDetection(self):
        if self.face_detector_model == "mediapipe":
            return MediaPipeWrapper()
        else:
            return MTCNNWrapper()

    def run(self):
        self.face_detector = self.FaceDetection()
        self.capture = cv.VideoCapture(f"rtsp://{self.username}:{self.password}@{self.IP}:554/stream1")
        fps = self.capture.get(cv.CAP_PROP_FPS)
        dropping = 1 if self.face_detector_model == "mediapipe" else 5
        recognizer_dropping = 5
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

            if frame_no%dropping == 0 or self.face_detector_model=="mediapipe":
                faces = self.face_detector.capture_faces(frame) 
            
            if not faces:
                continue
            bboxes = []
            
            for face in faces:
                (x,y),(w,h),(left_eye,right_eye) = self.face_detector.get_bbox(face,img_row,img_col)
                cv.rectangle(
                    frame, 
                    (x,y),
                    (w,h),
                    color = (0,255,0),
                    thickness=2
                )
                bboxes.append([x,y,w,h,left_eye,right_eye])

            self.stream_signal.emit(frame)
            """if frame_no%5==0:
                if self.MPqueue.full():
                    self.Mpqueue.get()
                    self.dropping += 2

                self.MPqueue.put((frame,bboxes))"""
            


    def stop(self):
        self.capture.release()
        self._run_flag=False
        self.wait()
