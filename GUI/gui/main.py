import sys
from PyQt5.QtGui import QKeyEvent, QImage, QPixmap
from PyQt5.QtWidgets import QListWidgetItem, QFileDialog, QMainWindow, QApplication, QMessageBox, QPushButton, QShortcut
from PyQt5.QtCore import pyqtSlot, Qt
from threading import Thread
from test import make_enc
import shutil
import os
import numpy as np
import cv2 as cv2
from gui_ui import Ui_MainWindow
from queue import Queue
from videostream import VideoStream
from facerecognition import FaceRecognition


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

        self.ui.upload_images_FL.addRow(QPushButton("BROWSE IMAGES", clicked = lambda: self.upload_images()))
        self.ui.upload_videos_FL.addRow(QPushButton("BROWSE VIDEOS", clicked = lambda: self.upload_videos()))
        self.ui.save_sett_btn.clicked.connect(self.save_settings)

        self.update_list('images')
        self.update_list('videos')
             

        self.queue = Queue(maxsize=1000)
        self.video_thread =VideoStream(
            queue = self.queue,
            username="aa2232786",
            password="aa2232786",
            IP="192.168.1.101",
            detection_model =self.get_detection_model()
        )        
        self.video_thread.stream_signal.connect(self.update_frame)
        self.video_thread.start()

        self.recog_thread = FaceRecognition(
            queue = self.queue,
            model_name = "Facenet"
        )
        self.recog_thread.start()
        


    @pyqtSlot(np.ndarray)
    def update_frame(self,frame):
        qt_frame = self.convert_cv_qt(frame)
        self.ui.home_page_label.setPixmap(qt_frame)
       
    def convert_cv_qt(self,rgb_frame):
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def get_detection_model(self):
        return self.ui.detection_model_index[self.ui.detection_model_CB.currentIndex()]

    def stop_video_stream_thread(self):
        self.video_thread.stop()


    def upload_images(self):
        # ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
        path = os.path.join(os.path.dirname(__file__), 'uploaded_images', self.ui.upload_images_text.text())
        try: os.mkdir(path)
        except OSError: pass
        img_paths, _ = QFileDialog.getOpenFileNames(None, "UPLOAD IMAGES", os.path.dirname(__file__), "Images (*.png *.jpg *.jpeg)")
        i = len(os.listdir(path))
        if img_paths:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                os.chdir(path)
                fname = self.ui.upload_images_text.text()+"_"+str(i)+"."+str(img_path.split('.')[-1])
                cv2.imwrite(fname, img)
                i+=1
            
            QMessageBox.information(self.ui.upload_images_page, 'Success', 'Images uploaded successfully!')
            self.ui.upload_images_text.clear()
            self.update_list(page='images')


    def upload_videos(self):
        

        self.recog_thread.stop()
        path = os.path.join(os.path.dirname(__file__), 'uploaded_videos', self.ui.upload_videos_text.text())

        if not os.path.isdir(path):
            os.makedirs(path)
        vid_paths, _ = QFileDialog.getOpenFileNames(None, "UPLOAD VIDEOS", os.path.dirname(__file__), "Videos (*.mp4)")
        i = len(os.listdir(path))

        if vid_paths:
            for vid_path in vid_paths:
                fname = self.ui.upload_videos_text.text()+"_"+str(i)+"."+str(vid_path.split('.')[-1])
                path = os.path.join(path, fname)
                # print(path)
                # print(vid_path)
                shutil.copy(vid_path, path)
                # print(path)
                t1 = Thread(target=make_enc,args=(path,))
                t1.start()
                t1.join()
                #make_enc(path+"/"+vid_path.split("/")[-1])
                '''capture = cv.VideoCapture(path+"/"+vid_path.split("/")[-1])
                fps = int(capture.get(cv.CAP_PROP_FPS))
                if fps==0:
                    fps = 15
                print(fps)
                frame_no = 0
                while(True):
                    isframe, frame = capture.read()

                    frame_no += 1
                    if frame_no%fps:
                        continue
                   
                    if not isframe:
                        break
                 
                    faces = face_detector.capture_faces(frame)
                    
                    if faces:
                        bbox = face_detector.get_bbox(faces[0],frame.shape[0],frame.shape[1])
                        
                        for face_model in face_models:
                            
                            face_model.create_save_encoding(
                                self.ui.upload_videos_text.text(),cv.cvtColor(frame,cv.COLOR_BGR2RGB),
                                bbox[0][0],bbox[0][1],
                                bbox[1][0],bbox[1][1],
                                bbox[2][0], bbox[2][1]
                            )

                capture.release()'''

            self.recog_thread = FaceRecognition(
                queue = self.queue,
                model_name = "Facenet"
            )
            self.recog_thread.start()
            QMessageBox.information(self.ui.upload_videos_page, 'Success', 'Videos uploaded successfully!')
            self.ui.upload_videos_text.clear()
            self.update_list(page='videos')

    def save_settings(self):
        print(self.ui.detection_model_CB.currentText().lower())
        print(self.ui.recognition_model_CB.currentText().lower())
        print(self.ui.processors_CB.currentText())

    # -----don't change this-----
    def update_list(self, page):
        if page == 'images':
            path = os.path.join(os.path.dirname(__file__), 'uploaded_images')
            self.ui.images_list_widget.clear()
        else:
            path = os.path.join(os.path.dirname(__file__), 'uploaded_videos')
            self.ui.videos_list_widget.clear()

        for i in os.listdir(path):
            inner_dir_path = os.path.join(path, i)
            if not os.path.isfile(inner_dir_path):
                for ele_path in os.listdir(inner_dir_path):
                    if page == 'images': QListWidgetItem(path+'\\'+i+'\\'+ele_path, self.ui.images_list_widget)
                    else: QListWidgetItem(path+'\\'+i+'\\'+ele_path, self.ui.videos_list_widget)
    def keyPressEvent(self, e: QKeyEvent):
        if self.ui.upload_videos_text.hasFocus() and e.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.upload_videos()
        if self.ui.upload_images_text.hasFocus() and e.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.upload_images()
    def toggleShadow(self, btn, shadow_obj):
        if btn.isChecked():
            shadow_obj.setEnabled(True)
        else:
            shadow_obj.setEnabled(False)
    def changeState(self, pressed):
        if pressed:
            self.ui.sidebar.show()
        else:
            self.ui.sidebar.hide()
    def on_home_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(0)
        self.toggleShadow(self.ui.home_btn, self.ui.home_btn_shadow)
    def on_show_data_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(1)
        self.toggleShadow(self.ui.show_data_btn, self.ui.show_data_btn_shadow)
    def on_recg_face_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(2)
        self.toggleShadow(self.ui.recg_face_btn, self.ui.recg_face_btn_shadow)
    def on_sett_panel_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(3)
        self.toggleShadow(self.ui.sett_panel_btn, self.ui.sett_panel_btn_shadow)
    def on_add_camera_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(4)
        self.toggleShadow(self.ui.add_camera_btn, self.ui.add_camera_btn_shadow)
    def on_upload_videos_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(5)
        self.toggleShadow(self.ui.upload_videos_btn, self.ui.upload_videos_btn_shadow)
    def on_upload_images_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(6)
        self.toggleShadow(self.ui.upload_images_btn, self.ui.upload_images_btn_shadow)
    # -------------------------
    
    def closeEvent(self, event):
        
        reply = QMessageBox.question(self, 'Message',
            "It may take a while, are you sure to quit?", QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.video_thread.requestInterruption()
            self.recog_thread.requestInterruption()
            self.video_thread.wait()
            self.recog_thread.wait()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    #loading style file
    with open("style.qss", "r") as style_file:
        style_str = style_file.read()
    
    app.setStyleSheet(style_str)


    window = MainWindow()
    window.show()

    sys.exit(app.exec())