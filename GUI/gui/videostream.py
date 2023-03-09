from FaceDetector.fd_mediaipipe import MediaPipeWrapper
from FaceDetector.fd_mtcnn import MTCNNWrapper
import numpy as np
import cv2 as cv
from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

class VideoStream(QThread):
    stream_signal = pyqtSignal(np.ndarray)

    def __init__(self, *arg, **kwargs) -> None:
        self.face_detector_model = kwargs["detection_model"]
        super(VideoStream, self).__init__()
        self.IP = kwargs["IP"]
        self.username = kwargs["username"]
        self.password = kwargs["password"]
        self.MPqueue = kwargs["queue"]
        self._run_flag = True
        self.clah = cv.createCLAHE(clipLimit=3.0,tileGridSize=(7,7))
        

    def FaceDetection(self):
        if self.face_detector_model == "mediapipe":
            return MediaPipeWrapper()
        else:
            return MTCNNWrapper()
        
    def frame_equlization_BGR_RGB(self,frame):
        frame = cv.cvtColor(frame,cv.COLOR_BGR2LAB)

        frame[:,:,0] = self.clah.apply(frame[:,:,0])

        return cv.cvtColor(frame,cv.COLOR_LAB2RGB)
    


    def run(self):
        self.face_detector = self.FaceDetection()
        self.capture = cv.VideoCapture(
            f"rtsp://{self.username}:{self.password}@{self.IP}:554/stream1"
        )
        fps = self.capture.get(cv.CAP_PROP_FPS)
        print(fps)
        dropping = 1 if self.face_detector_model == "mediapipe" else 5
        recognizer_dropping = 1
        frame_no = 0
        faces = None
        while not self.isInterruptionRequested():
            isframe, frame = self.capture.read()
            if not isframe:
                continue
            curr_time = datetime.now()

            frame_no += 1
            if frame_no % fps == 0:
                frame_no = 0

            img_row, img_col = frame.shape[0], frame.shape[1]
            frame = self.frame_equlization_BGR_RGB(frame)

            if frame_no % dropping == 0 or self.face_detector_model == "mediapipe":
                faces = self.face_detector.capture_faces(frame)
           
            
            bboxes = []
            if faces:
                for face in faces:
                    (x, y), (w, h), (left_eye, right_eye) = self.face_detector.get_bbox(
                        face, img_row, img_col
                    )
                    cv.rectangle(frame, (x, y), (w, h), color=(0, 255, 0), thickness=2)
                    bboxes.append(
                        {
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "left_eye": left_eye,
                            "right_eye": right_eye,
                        }
                    )

            self.stream_signal.emit(frame)
            if frame_no%recognizer_dropping==0:
                if self.MPqueue.full():
                    self.MPqueue.get()
                    recognizer_dropping += 2

                self.MPqueue.put(
                    (frame, bboxes, curr_time.strftime("%H:%M:%S"))
                )

    def stop(self):
        self.capture.release()
        self._run_flag = False
        self.wait()
