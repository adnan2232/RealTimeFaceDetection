import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import pyqtSlot, Qt
import numpy as np
import cv2 as cv
from gui_ui import Ui_MainWindow
from multiprocessing import Queue
from videostream import VideoStream

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
            IP="192.168.1.106",
            detection_model = self.get_detection_model()
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

    def get_detection_model(self):
        return self.ui.detection_model_index[self.ui.detection_model_CB.currentIndex()]

    def stop_video_stream_thread(self):
        self.video_thread.stop()

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