import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream, QThread, pyqtSignal, Qt
import numpy as np
import cv2 as cv
from mtcnn_cv2 import MTCNN
from gui_ui import Ui_MainWindow
from joblib import Parallel,delayed
from multiprocessing import Queue
import mediapipe as mp
class VideoStream(QThread):
    stream_signal = pyqtSignal(np.ndarray)

    def __init__(self,MPqueue,*arg,**kwargs) -> None:
        self.face_detector_model = "mtcnn"
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
            color = (255,0,0),
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



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.menu_btn.clicked[bool].connect(self.changeState)
        self.display_height = 590
        self.display_width = 733
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.sidebar.hide()

        self.ui.detection_model_CB.activated.connect(self.foo1)
        self.ui.recognition_model_CB.activated.connect(self.foo2)
        self.ui.processors_CB.activated.connect(self.foo3)

        self.MPQueue = Queue()
        self.video_thread =VideoStream(
            self.MPQueue,
            username="aa2232786",
            password="aa2232786",
            IP="192.168.1.103"
        )        
        self.video_thread.stream_signal.connect(self.update_frame)
        self.video_thread.start()
    
    @pyqtSlot(np.ndarray)
    def update_frame(self,frame):
        qt_frame = self.convert_cv_qt(frame)
        self.ui.label_2.setPixmap(qt_frame)
       
    def convert_cv_qt(self,rgb_frame):
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    def changeState(self, pressed):
        if pressed:
            self.ui.sidebar.show()
        else:
            self.ui.sidebar.hide()
    
    def on_home_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(0)
    
    def on_add_data_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(1)
        
    def on_recg_face_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def on_sett_panel_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(3)

    def on_add_camera_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(4)

    def on_add_fp_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(5)
    
    def foo1(self):
        # self.video_thread.face_detector_model = self.ui.detection_model_CB.currentText().lower()
        print(self.ui.detection_model_CB.currentText().lower())
    
    def foo2(self):
        # self.video_thread.face_recognizer_model = self.ui.recognition_model_CB.currentText().lower()
        print(self.ui.recognition_model_CB.currentText().lower())
    
    def foo3(self):
        # self.video_thread.processors = self.ui.processors_CB.currentText()
        print(self.ui.processors_CB.currentText())



if __name__ == "__main__":
    app = QApplication(sys.argv)

    #loading style file
    with open("style.qss", "r") as style_file:
        style_str = style_file.read()
    
    app.setStyleSheet(style_str)


    window = MainWindow()
    window.show()

    sys.exit(app.exec())