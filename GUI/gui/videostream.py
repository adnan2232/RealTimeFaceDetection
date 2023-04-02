from FaceDetector.fd_mediaipipe import MediaPipeWrapper
from FaceDetector.fd_mtcnn import MTCNNWrapper
from collections import deque
from math import ceil
import numpy as np
import cv2 as cv
from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime
from time import monotonic
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

class VideoStream(QThread):
    stream_signal = pyqtSignal(np.ndarray)

    def __init__(self, *arg, **kwargs) -> None:
        self.face_detector_model = kwargs["detection_model"]
        super(VideoStream, self).__init__()
        self.camera_info = kwargs["camera_info"] if kwargs['camera_info']!='0' else int(kwargs['camera_info'])
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
        try:
            self.capture = cv.VideoCapture(self.camera_info)
      
            
            # self.capture = cv.VideoCapture(0)
            fps = self.capture.get(cv.CAP_PROP_FPS)
            if fps==0:
                fps = 15
    
            detect_time = deque(maxlen=10)

            detector_dropping = 1 
            recognize_dropping=1
        
        
            frame_no = 0
            faces = None
            while not self.isInterruptionRequested():
                isframe, frame = self.capture.read()

                if not isframe:
                    continue
                curr_time = datetime.now()

                frame_no += 1
                if frame_no%fps==0:
                    frame_no = 0

                img_row, img_col = frame.shape[0], frame.shape[1]
                frame = self.frame_equlization_BGR_RGB(frame)

                if frame_no % detector_dropping == 0:
                    start_time = monotonic()
                    faces = self.face_detector.capture_faces(frame)
                    end_time = monotonic()
                    detect_time.append((end_time-start_time)*fps)

                med = np.median(detect_time)
            
                if len(detect_time)>5 and not np.isnan(med) and ceil(med)> detector_dropping:
                    print(detect_time)
                    detector_dropping = min(10,ceil(med))
                    detect_time.clear()
                    print(detector_dropping)
                elif len(detect_time)>5 and not np.isnan(med) and detector_dropping-ceil(med) >1:
                    print(detect_time)
                    detector_dropping = max(1,ceil(med))
                    print(detector_dropping)

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

                self.stream_signal.emit(frame.copy())
                curr_qsize = self.MPqueue.qsize()
                
                if curr_qsize-50*(recognize_dropping-1) >=50:
                    recognize_dropping = min(10,recognize_dropping+1)
                elif 50*recognize_dropping-curr_qsize >= 50:
                    recognize_dropping = max(1,recognize_dropping-1)

                if faces and frame_no%recognize_dropping==0:
                    
                    # print(f"qsize: {curr_qsize}, dropping:{recognize_dropping}")
                    self.MPqueue.put(
                        (frame.copy(), bboxes.copy(), curr_time.strftime("%H:%M:%S"))
                    )
                del frame
                del bboxes
        except:
            pass
        finally:
            self.stop()

    def stop(self):
        self.capture.release()
        self._run_flag = False
       
        
        
        
        
        
