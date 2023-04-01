import sys
from PyQt5.QtGui import QKeyEvent, QImage, QPixmap
from PyQt5.QtWidgets import QListWidgetItem, QFileDialog, QMainWindow, QApplication, QMessageBox, QPushButton, QShortcut
from PyQt5.QtCore import pyqtSlot, Qt
from threading import Thread
from store_encoding import make_enc
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
        self.ui.add_camera_FL.addRow(QPushButton("ADD CAMERA", clicked = lambda: self.add_camera()))
        self.ui.save_sett_btn.clicked.connect(self.save_settings)
        self.ui.images_LW.itemClicked.connect(self.images_clicked_LW)
        self.ui.videos_LW.itemClicked.connect(self.videos_clicked_LW)
             
        self.queue = Queue(maxsize=1000)
        self.start_video_thread()
        self.start_recog_thread()
  
    def images_clicked_LW(self, item):

        if item.text() == '----NO DATA TO SHOW----':
            return

        folder = os.path.join(os.path.dirname(__file__), 'uploaded_images', item.text())
        filename = os.path.join(folder, os.listdir(folder)[0])

        self.ui.image_lbl.setPixmap(QPixmap(filename))
        self.ui.profile_sec_lbl.show()
        self.ui.profile_sec.show()
        self.ui.name.setText(item.text())

        # get encoding generated time here
        self.ui.enc_time.setText('--time--')

        # link delete encoding method here
        # self.ui.del_btn.clicked.connect()

    def videos_clicked_LW(self, item):

        if item.text() == '----NO DATA TO SHOW----':
            return

        folder = os.path.join(os.path.dirname(__file__), 'uploaded_videos', item.text())
        filename = os.path.join(folder, os.listdir(folder)[0])

        # extracting first frame
        os.chdir(folder)
        if 'first_frame.jpg' not in os.listdir(folder):
            cap = cv2.VideoCapture(filename)
            success, image = cap.read()
            if success:
                cv2.imwrite('first_frame.jpg', image)

        self.ui.image_lbl.setPixmap(QPixmap('first_frame.jpg'))
        self.ui.profile_sec_lbl.show()
        self.ui.profile_sec.show()
        self.ui.name.setText(item.text())

        # get encoding generated time here
        self.ui.enc_time.setText('--time--')

        # link delete encoding method here
        # self.ui.del_btn.clicked.connect()


        
 
    def start_video_thread(self):
        self.detection_model_name = self.get_detection_model()
        self.video_thread =VideoStream(
            queue = self.queue,
            username="aa2232786",
            password="aa2232786",
            IP="192.168.1.102",
            detection_model = self.detection_model_name
        ) 
        self.video_thread.stream_signal.connect(self.update_frame)
        self.video_thread.start()
       

    def start_recog_thread(self):
      
        self.recog_model_name = self.get_recog_model()
        self.recog_thread = FaceRecognition(
            queue = self.queue,
            model_name = self.recog_model_name
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
    
    def get_recog_model(self):
        return self.ui.recognition_model_index[self.ui.recognition_model_CB.currentIndex()]

    def stop_video_thread(self):
        
        self.video_thread.requestInterruption()

    def stop_recog_thread(self,empty_q:bool=False):

        self.recog_thread.stop(empty_q)

    
    def change_detection_model(self):

        self.stop_video_thread()
        self.start_video_thread()

    def change_recog_model(self):
        self.recog_model_name =self.get_recog_model()
        self.stop_recog_thread(False)
        self.start_recog_thread()

    def save_settings(self):
        if self.detection_model_name != self.get_detection_model():
            self.change_detection_model()
        if self.recog_model_name != self.get_recog_model():
            self.change_recog_model()

    def upload_images(self):
        
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
        
        if self.recog_thread.isRunning():
            self.stop_recog_thread(False)

        name = self.ui.upload_videos_text.text()
        path = os.path.join(os.path.dirname(__file__), 'uploaded_videos',name )

        if not os.path.isdir(path):
            os.makedirs(path)
        vid_paths, _ = QFileDialog.getOpenFileNames(None, "UPLOAD VIDEOS", os.path.dirname(__file__), "Videos (*.mp4)")
        
        file_list = os.listdir(path)
        if 'first_frame.jpg' in file_list:
            file_list.remove('first_frame.jpg')
        i = len(file_list)

        if vid_paths:
            for vid_path in vid_paths:
                fname = name+"_"+str(i)+"."+str(vid_path.split('.')[-1])
                path = os.path.join(path, fname)
                shutil.copy(vid_path, path)
                t1 = Thread(target=make_enc,args=(path,name))
                t1.start()
                t1.join()
                
            self.start_recog_thread()
          
            QMessageBox.information(self.ui.upload_videos_page, 'Success', 'Videos uploaded successfully!')
            self.ui.upload_videos_text.clear()
            self.update_list(page='videos')

    def add_camera(self):
        #do something



        # success msg
        QMessageBox.information(self.ui.add_camera_page, 'Success', 'Camera added successfully!')
        self.ui.add_camera_text.clear()



    # -----don't change this-----
    def update_list(self, page):

        if page == 'images':
            path = os.path.join(os.path.dirname(__file__), 'uploaded_images')
            self.ui.images_list_widget.clear()
        else:
            path = os.path.join(os.path.dirname(__file__), 'uploaded_videos')
            self.ui.videos_list_widget.clear()
        if not os.path.isdir(path):
            os.mkdir(path)
        if len(os.listdir(path)) == 0:
            if page == 'images': QListWidgetItem('----NO DATA TO SHOW----', self.ui.images_list_widget)
            else: QListWidgetItem('----NO DATA TO SHOW----', self.ui.videos_list_widget)
            return
        for i in os.listdir(path):
            inner_dir_path = os.path.join(path, i)
            if not os.path.isfile(inner_dir_path):
                file_list = os.listdir(inner_dir_path)
                if 'first_frame.jpg' in file_list:
                    file_list.remove('first_frame.jpg')
                for ele_path in file_list:
                    if page == 'images': QListWidgetItem(path+'\\'+i+'\\'+ele_path, self.ui.images_list_widget)
                    else: QListWidgetItem(path+'\\'+i+'\\'+ele_path, self.ui.videos_list_widget)

    def show_data_list(self, page):
        
        if page == 'images':
            path = os.path.join(os.path.dirname(__file__), 'uploaded_images')
            self.ui.images_LW.clear()
        else:
            path = os.path.join(os.path.dirname(__file__), 'uploaded_videos')
            self.ui.videos_LW.clear()
        if not os.path.isdir(path):
            os.mkdir(path)
        if len(os.listdir(path))==0:
            if page == 'images': QListWidgetItem('----NO DATA TO SHOW----', self.ui.images_LW)
            else: QListWidgetItem('----NO DATA TO SHOW----', self.ui.videos_LW)
            return
        for i in os.listdir(path):
            if page == 'images': 
                self.ui.images_LW.addItem(i)
            else: 
                self.ui.videos_LW.addItem(i)
        
    def keyPressEvent(self, e: QKeyEvent):
        if self.ui.upload_videos_text.hasFocus() and e.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.upload_videos()
        if self.ui.upload_images_text.hasFocus() and e.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.upload_images()
        if self.ui.add_camera_text.hasFocus() and e.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.add_camera()
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
    def on_upload_images_btn_toggled(self):
        self.update_list('images')
        self.ui.stackedWidget.setCurrentIndex(1)
        self.toggleShadow(self.ui.upload_images_btn, self.ui.upload_images_btn_shadow)
    def on_upload_videos_btn_toggled(self):
        self.update_list('videos')
        self.ui.stackedWidget.setCurrentIndex(2)
        self.toggleShadow(self.ui.upload_videos_btn, self.ui.upload_videos_btn_shadow)
    def on_show_data_btn_toggled(self):
        self.show_data_list('images')
        self.show_data_list('videos')
        self.ui.profile_sec_lbl.hide()
        self.ui.profile_sec.hide()
        self.ui.stackedWidget.setCurrentIndex(3)
        self.toggleShadow(self.ui.show_data_btn, self.ui.show_data_btn_shadow)
    def on_add_camera_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(4)
        self.toggleShadow(self.ui.add_camera_btn, self.ui.add_camera_btn_shadow)
    def on_sett_panel_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(5)
        self.toggleShadow(self.ui.sett_panel_btn, self.ui.sett_panel_btn_shadow)
    def on_recg_face_btn_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(6)
        self.toggleShadow(self.ui.recg_face_btn, self.ui.recg_face_btn_shadow)
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