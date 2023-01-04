import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream, QThread, pyqtSignal, Qt
import numpy as np
import cv2 as cv
from mtcnn import MTCNN
from gui_ui import Ui_MainWindow
from joblib import Parallel,delayed
from multiprocessing import Queue

class VideoStream(QThread):
    stream_signal = pyqtSignal(np.ndarray)

    def __init__(self,MPqueue,*arg,**kwargs) -> None:
        super(VideoStream,self).__init__()
        self.IP = kwargs["IP"]
        self.username = kwargs["username"]
        self.password = kwargs["password"]
        self.MPqueue = MPqueue

    def send_queue(self,frame,face):
        x,y,w,h = face['box']
        self.MPqueue.put(frame[y:y+h,x:x+w])
        cv.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=1) 

    def run(self):
        face_detector = MTCNN()
        
        capture = cv.VideoCapture(f"rtsp://{self.username}:{self.password}@{self.IP}:554/stream2")
        while(True):
            isFrame, Frame = capture.read()
            if not isFrame:
                continue
            faces = face_detector.detect_faces(
                cv.cvtColor(Frame,cv.COLOR_BGR2RGB)
            )

            Parallel(n_jobs=-1,prefer="threads")(delayed(self.send_queue)(face,Frame) for face in faces)
            self.stream_signal.emit(Frame)
            if not self.MPqueue.empty():
                self.MPqueue.get()



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.display_width = 640
        self.display_height = 480
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.menu_btn.clicked[bool].connect(self.changeState)

        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.sidebar.hide()
        self.MPQueue = Queue()
        self.video_thread =VideoStream(
            self.MPQueue,
            username="aa2232786",
            password="aa2232786",
            IP="192.168.1.105"
        )        
        self.video_thread.stream_signal.connect(self.update_frame)
        self.video_thread.start()
    
    @pyqtSlot(np.ndarray)
    def update_frame(self,frame):
        qt_frame = self.convert_cv_qt(frame)
       

    def convert_cv_qt(self,frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    def changeState(self, pressed):
        # print(pressed)
        if pressed:
            self.ui.sidebar.show()
        else:
            self.ui.sidebar.hide()

    # def on_stackedWidget_currentChanged(self, index):
    #     btn_list = self.ui.sidebar.findChildren(QPushButton)
    #     print(index)

    #     for btn in btn_list:
    #         if index in [5, 6]:
    #             btn.setAutoExclusive(False)
    #             btn.setChecked(False)
    #             print("if")
    #         else:
    #             btn.setAutoExclusive(True)
    #             print("else")
    
    
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



if __name__ == "__main__":
    app = QApplication(sys.argv)

    #loading style file
    with open("style.qss", "r") as style_file:
        style_str = style_file.read()
    
    app.setStyleSheet(style_str)


    window = MainWindow()
    window.show()

    sys.exit(app.exec())