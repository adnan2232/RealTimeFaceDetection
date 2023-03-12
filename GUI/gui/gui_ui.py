# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog

import os
import cv2
import datetime
import shutil


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        # Root Dir
        self.ROOT_DIR = os.path.dirname(__file__)
        # self.first_run = True

        # main window
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(927, 672)

        # font settings
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)

        # central widget 
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setFont(font)
        self.centralwidget.setObjectName("centralwidget")


        # --------------------header widget start--------------------

        # header widget
        self.header = QtWidgets.QWidget(self.centralwidget)
        self.header.setObjectName("header")

        # menu button
        self.menu_btn = QtWidgets.QPushButton(self.header)
        self.menu_btn.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon/icons8-circled-menu-50.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menu_btn.setIcon(icon)
        self.menu_btn.setIconSize(QtCore.QSize(20, 20))
        self.menu_btn.setCheckable(True)
        self.menu_btn.setChecked(False)
        # self.menu_btn.setChecked(True)
        self.menu_btn.setAutoExclusive(True)
        self.menu_btn.setObjectName("menu_btn")

        # vertical spacer between menu button and face recognition label
        spacerItem = QtWidgets.QSpacerItem(369, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        # face recognition label
        self.fr_label = QtWidgets.QLabel(self.header)
        self.fr_label.setScaledContents(True)
        self.fr_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fr_label.setObjectName("fr_label")

        # vertical spacer between face recognition label and end of screen
        spacerItem1 = QtWidgets.QSpacerItem(369, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        # header widget in horizontal layout
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.header)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout_2.addWidget(self.menu_btn)
        self.horizontalLayout_2.addItem(spacerItem)
        self.horizontalLayout_2.addWidget(self.fr_label)
        self.horizontalLayout_2.addItem(spacerItem1)


        # --------------------header widget end--------------------



        # --------------------sidebar widget start--------------------

        # sidebar widget
        self.sidebar = QtWidgets.QWidget(self.centralwidget)
        self.sidebar.setObjectName("sidebar")

        # shadow for every button
        # self.sideBar_btns_shadow = QtWidgets.QGraphicsDropShadowEffect()
        # self.sideBar_btns_shadow.setBlurRadius(15) 

        # home button
        self.home_btn = QtWidgets.QPushButton(self.sidebar)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icon/home.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.home_btn.setIcon(icon1)
        self.home_btn.setIconSize(QtCore.QSize(20, 20))
        self.home_btn.setCheckable(True)
        self.home_btn.setAutoExclusive(True)
        self.home_btn.setObjectName("home_btn")
        self.home_btn_shadow = QtWidgets.QGraphicsDropShadowEffect()
        self.home_btn_shadow.setBlurRadius(15)
        self.home_btn.setGraphicsEffect(self.home_btn_shadow)
        self.home_btn_shadow.setEnabled(False)

        # upload images button
        self.upload_images_btn = QtWidgets.QPushButton(self.sidebar)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("icon/add-contact.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.upload_images_btn.setIcon(icon6)
        self.upload_images_btn.setIconSize(QtCore.QSize(20, 20))
        self.upload_images_btn.setCheckable(True)
        self.upload_images_btn.setAutoExclusive(True)
        self.upload_images_btn.setObjectName("upload_images_btn")
        self.upload_images_btn_shadow = QtWidgets.QGraphicsDropShadowEffect()
        self.upload_images_btn_shadow.setBlurRadius(15)
        self.upload_images_btn.setGraphicsEffect(self.upload_images_btn_shadow)
        self.upload_images_btn_shadow.setEnabled(False)

        # upload videos button
        self.upload_videos_btn = QtWidgets.QPushButton(self.sidebar)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("icon/add-file-path2.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.upload_videos_btn.setIcon(icon6)
        self.upload_videos_btn.setIconSize(QtCore.QSize(20, 20))
        self.upload_videos_btn.setCheckable(True)
        self.upload_videos_btn.setAutoExclusive(True)
        self.upload_videos_btn.setObjectName("upload_videos_btn")
        self.upload_videos_btn_shadow = QtWidgets.QGraphicsDropShadowEffect()
        self.upload_videos_btn_shadow.setBlurRadius(15)
        self.upload_videos_btn.setGraphicsEffect(self.upload_videos_btn_shadow)
        self.upload_videos_btn_shadow.setEnabled(False)

        # show data button
        self.show_data_btn = QtWidgets.QPushButton(self.sidebar)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icon/add-database-32.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.show_data_btn.setIcon(icon2)
        self.show_data_btn.setIconSize(QtCore.QSize(20, 20))
        self.show_data_btn.setCheckable(True)
        self.show_data_btn.setAutoExclusive(True)
        self.show_data_btn.setObjectName("show_data_btn")
        self.show_data_btn_shadow = QtWidgets.QGraphicsDropShadowEffect()
        self.show_data_btn_shadow.setBlurRadius(15)
        self.show_data_btn.setGraphicsEffect(self.show_data_btn_shadow)
        self.show_data_btn_shadow.setEnabled(False)

        # add camera button
        self.add_camera_btn = QtWidgets.QPushButton(self.sidebar)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("icon/add-photo-camera1.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.add_camera_btn.setIcon(icon5)
        self.add_camera_btn.setIconSize(QtCore.QSize(20, 20))
        self.add_camera_btn.setCheckable(True)
        self.add_camera_btn.setAutoExclusive(True)
        self.add_camera_btn.setObjectName("add_camera_btn")
        self.add_camera_btn_shadow = QtWidgets.QGraphicsDropShadowEffect()
        self.add_camera_btn_shadow.setBlurRadius(15)
        self.add_camera_btn.setGraphicsEffect(self.add_camera_btn_shadow)
        self.add_camera_btn_shadow.setEnabled(False)

        # settings panel button
        self.sett_panel_btn = QtWidgets.QPushButton(self.sidebar)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("icon/settings-17-32.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.sett_panel_btn.setIcon(icon4)
        self.sett_panel_btn.setIconSize(QtCore.QSize(20, 20))
        self.sett_panel_btn.setCheckable(True)
        self.sett_panel_btn.setAutoExclusive(True)
        self.sett_panel_btn.setObjectName("sett_panel_btn")
        self.sett_panel_btn_shadow = QtWidgets.QGraphicsDropShadowEffect()
        self.sett_panel_btn_shadow.setBlurRadius(15)
        self.sett_panel_btn.setGraphicsEffect(self.sett_panel_btn_shadow)
        self.sett_panel_btn_shadow.setEnabled(False)

        # recognize face button
        self.recg_face_btn = QtWidgets.QPushButton(self.sidebar)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("icon/Face_Recognize13.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.recg_face_btn.setIcon(icon3)
        self.recg_face_btn.setIconSize(QtCore.QSize(20, 20))
        self.recg_face_btn.setCheckable(True)
        self.recg_face_btn.setAutoExclusive(True)
        self.recg_face_btn.setObjectName("recg_face_btn")
        self.recg_face_btn_shadow = QtWidgets.QGraphicsDropShadowEffect()
        self.recg_face_btn_shadow.setBlurRadius(15)
        self.recg_face_btn.setGraphicsEffect(self.recg_face_btn_shadow)
        self.recg_face_btn_shadow.setEnabled(False)
        
        # above buttons in vertical layout
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.addWidget(self.home_btn)
        self.verticalLayout.addWidget(self.upload_images_btn)
        self.verticalLayout.addWidget(self.upload_videos_btn)
        self.verticalLayout.addWidget(self.show_data_btn)
        self.verticalLayout.addWidget(self.add_camera_btn)
        self.verticalLayout.addWidget(self.sett_panel_btn)
        self.verticalLayout.addWidget(self.recg_face_btn)

        # vertical spacer between vertical layout of buttons and close button
        spacerItem2 = QtWidgets.QSpacerItem(20, 399, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        # close button
        self.close_btn = QtWidgets.QPushButton(self.sidebar)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("icon/icons8-cancel-50.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.close_btn.setIcon(icon7)
        self.close_btn.setIconSize(QtCore.QSize(20, 20))
        self.close_btn.setCheckable(True)
        self.close_btn.setAutoExclusive(True)
        self.close_btn.setObjectName("close_btn")
        
        # set of buttons, vertical spacer and close button in vertical layout
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.sidebar)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2.addItem(spacerItem2)
        self.verticalLayout_2.addWidget(self.close_btn)


        # --------------------sidebar widget end--------------------



        # --------------------content page widget start--------------------        

        # content page
        self.content_page = QtWidgets.QWidget(self.centralwidget)
        self.content_page.setObjectName("content_page")

        # stacked widget
        self.stackedWidget = QtWidgets.QStackedWidget(self.content_page)
        self.stackedWidget.setObjectName("stackedWidget")


        # --------------------home page start--------------------

        # home page
        self.home_page = QtWidgets.QWidget()
        self.home_page.setObjectName("home_page")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.home_page)
        self.gridLayout_2.setObjectName("gridLayout_2")
        # 4 spacers
        # HP = home panel, T = top, L = left, R = right, B = bottom, S = spacer
        HPLS = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        HPTS = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        HPRS = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        HPBS = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        # home page in label
        self.home_page_label = QtWidgets.QLabel(self.home_page)
        self.home_page_label.setAlignment(QtCore.Qt.AlignCenter)
        self.home_page_label.setObjectName("home_page_label")
        
        self.gridLayout_2.addItem(HPTS, 0, 1, 1, 1)
        self.gridLayout_2.addItem(HPLS, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.home_page_label, 1, 1, 1, 2)
        self.gridLayout_2.addItem(HPRS, 1, 3, 1, 1)
        self.gridLayout_2.addItem(HPBS, 2, 2, 1, 1)
        # self.gridLayout_2.addWidget(self.home_page_label, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.home_page)


        # --------------------home page page end--------------------



        # --------------------show data page start--------------------

        # add data page
        self.show_data_page = QtWidgets.QWidget()
        self.show_data_page.setObjectName("show_data_page")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.show_data_page)
        self.gridLayout_3.setObjectName("gridLayout_3")
        # add data label
        self.label_3 = QtWidgets.QLabel(self.show_data_page)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.show_data_page)


        # --------------------show data page end--------------------


        # --------------------recognize face page start--------------------

        # recognize face page
        self.recg_face_page = QtWidgets.QWidget()
        self.recg_face_page.setObjectName("recg_face_page")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.recg_face_page)
        self.gridLayout_4.setObjectName("gridLayout_4")
        # recognize face label
        self.label_4 = QtWidgets.QLabel(self.recg_face_page)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.recg_face_page)


        # --------------------recognize face page end--------------------



        # --------------------settings panel page start--------------------

        # settings panel page
        self.sett_panel_page = QtWidgets.QWidget()
        self.sett_panel_page.setObjectName("sett_panel_page")


        # settings panel label
        self.sett_panel_label = QtWidgets.QLabel(self.sett_panel_page)
        self.sett_panel_label.setScaledContents(True)
        self.sett_panel_label.setAlignment(QtCore.Qt.AlignCenter)
        self.sett_panel_label.setObjectName("sett_panel_label")

        # 4 spacers
        # SP = settings panel, T = top, L = left, R = right, B = bottom, S = spacer
        SPTS = QtWidgets.QSpacerItem(20, 133, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        SPLS = QtWidgets.QSpacerItem(71, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        SPRS = QtWidgets.QSpacerItem(70, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        SPBS = QtWidgets.QSpacerItem(20, 133, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        # detection model label
        self.detection_model_label = QtWidgets.QLabel(self.sett_panel_page)
        self.detection_model_label.setMinimumSize(QtCore.QSize(220, 25))
        self.detection_model_label.setScaledContents(True)
        self.detection_model_label.setAlignment(QtCore.Qt.AlignCenter)
        self.detection_model_label.setObjectName("detection_model_label")
        # detection model combobox
        self.detection_model_CB = QtWidgets.QComboBox(self.sett_panel_page)
        self.detection_model_CB.setMinimumSize(QtCore.QSize(200, 20))
        self.detection_model_CB.setObjectName("detection_model_CB")
        self.detection_model_index = {0:"MTCNN",1:"mediapipe"}
        self.detection_model_CB.addItems(['MTCNN', 'mediapipe'])
        self.detection_model_CB.setCurrentIndex(1)
        # horizontal spacer between detection model label and detection model combobox
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # detection model cell in horizontal layout
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.addWidget(self.detection_model_label)
        self.horizontalLayout.addItem(spacerItem5)
        self.horizontalLayout.addWidget(self.detection_model_CB)

        # vertical spacer between detection model cell and recognition model cell
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        # recognition model label
        self.recognition_model_label = QtWidgets.QLabel(self.sett_panel_page)
        self.recognition_model_label.setMinimumSize(QtCore.QSize(220, 25))
        self.recognition_model_label.setScaledContents(True)
        self.recognition_model_label.setAlignment(QtCore.Qt.AlignCenter)
        self.recognition_model_label.setObjectName("recognition_model_label")
        # recognition model combobox
        self.recognition_model_CB = QtWidgets.QComboBox(self.sett_panel_page)
        self.recognition_model_CB.setMinimumSize(QtCore.QSize(200, 20))
        self.recognition_model_CB.setObjectName("recognition_model_CB")
        self.recognition_model_CB.addItems(['FaceNet512', 'ArcFace'])
        # horizontal spacer between recognition model label and recognition model combobox
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # recognition model cell in horizontal layout
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_3.addWidget(self.recognition_model_label)
        self.horizontalLayout_3.addItem(spacerItem7)
        self.horizontalLayout_3.addWidget(self.recognition_model_CB)

        # vertical spacer between recognition model cell and processors cell
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        # processor label
        self.processors_label = QtWidgets.QLabel(self.sett_panel_page)
        self.processors_label.setMinimumSize(QtCore.QSize(220, 25))
        self.processors_label.setScaledContents(True)
        self.processors_label.setAlignment(QtCore.Qt.AlignCenter)
        self.processors_label.setObjectName("processors_label")
        # processor combobox 
        self.processors_CB = QtWidgets.QComboBox(self.sett_panel_page)
        self.processors_CB.setMinimumSize(QtCore.QSize(200, 20))
        self.processors_CB.setObjectName("processors_CB")
        # add number of processors
        cpu_count = os.cpu_count()
        processors = []
        for i in range(2, cpu_count+1):
            processors.append(str(i))
        self.processors_CB.addItems(processors)
        # horizontal spacer between processor label and processor combobox
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # processors cell in horizontal layout
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_4.addWidget(self.processors_label)
        self.horizontalLayout_4.addItem(spacerItem9)
        self.horizontalLayout_4.addWidget(self.processors_CB)

        # vertical spacer between processors cell and save settings button
        processorsCellSaveBtnCellSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        
        # save settings button
        self.save_sett_btn = QtWidgets.QPushButton(self.sett_panel_page)
        self.save_sett_btn.setObjectName("save_sett_btn")

        # detection model cell, recognition model cell, processors cell, 
        # save settings button and vertical spacers in vertical layout
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(50, 50, 50, 50)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout_3.addItem(spacerItem6)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3.addItem(spacerItem8)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.verticalLayout_3.addItem(processorsCellSaveBtnCellSpacer)
        self.verticalLayout_3.addWidget(self.save_sett_btn)

        # all elements of settings panel in grid layout
        self.settPanelGL = QtWidgets.QGridLayout(self.sett_panel_page)
        self.settPanelGL.setObjectName("settPanelGL")
        self.settPanelGL.addWidget(self.sett_panel_label, 0, 1, 1, 2)
        self.settPanelGL.addItem(SPTS, 1, 1, 1, 1)
        self.settPanelGL.addItem(SPLS, 2, 0, 1, 1)
        self.settPanelGL.addLayout(self.verticalLayout_3, 2, 1, 1, 2)
        self.settPanelGL.addItem(SPRS, 2, 3, 1, 1)
        self.settPanelGL.addItem(SPBS, 3, 2, 1, 1)
        
        self.stackedWidget.addWidget(self.sett_panel_page)


        # --------------------settings panel page end--------------------



        # --------------------add camera page start--------------------

        # add camera page
        self.add_camera_page = QtWidgets.QWidget()
        self.add_camera_page.setObjectName("add_camera_page")

        # add camera form
        self.add_camera_FL = QtWidgets.QFormLayout()
        self.add_camera_FL.addRow(QtWidgets.QLabel("ADD CAMERA LINK: "), QtWidgets.QLineEdit())

        self.add_camera_page.setLayout(self.add_camera_FL)

        self.stackedWidget.addWidget(self.add_camera_page)

        # --------------------add camera page end--------------------



        # --------------------upload videos page start--------------------

        # test video page
        self.upload_videos_page = QtWidgets.QWidget()
        self.upload_videos_page.setObjectName("upload_videos_page")

        # test video form
        self.upload_videos_FL = QtWidgets.QFormLayout()
        self.upload_videos_text = QtWidgets.QLineEdit()

        # adding rows to form layout
        self.upload_videos_FL.addRow(QtWidgets.QLabel("ADD FOLDER NAME: "), self.upload_videos_text)
        self.upload_videos_FL.addRow(QtWidgets.QPushButton("BROWSE VIDEOS", clicked = lambda: self.upload_videos()))


        
        # test video page layout
        self.upload_videos_vbl = QtWidgets.QVBoxLayout()
        self.upload_videos_vbl.addLayout(self.upload_videos_FL)

        self.upload_videos_page.setLayout(self.upload_videos_vbl)

        # test video page in stacked widget
        self.stackedWidget.addWidget(self.upload_videos_page)
        
        # --------------------upload videos page end--------------------



        # --------------------upload images page start--------------------

        # upload faces page
        self.upload_images_page = QtWidgets.QWidget()
        self.upload_images_page.setObjectName("upload_images_page")

        self.vbl = QtWidgets.QVBoxLayout()
        self.upload_face_vbl = QtWidgets.QVBoxLayout()
        self.uploads_vbl = QtWidgets.QVBoxLayout()

        # successful message label
        self.success_label = QtWidgets.QLabel(self.upload_images_page)
        self.success_label.setMinimumSize(QtCore.QSize(500, 20))
        self.success_label.setAlignment(QtCore.Qt.AlignCenter)
        self.success_label.setObjectName("success_label")
        # successful message close button
        self.success_label_btn = QtWidgets.QPushButton(self.upload_images_page)
        self.success_label_btn.setIcon(icon7)
        self.success_label_btn.setIconSize(QtCore.QSize(20, 20))
        self.success_label_btn.setCheckable(True)
        self.success_label_btn.setAutoExclusive(True)
        self.success_label_btn.setObjectName("success_label_btn")
        self.success_label_btn.clicked.connect(
            lambda: self.success_close(self.success_label, self.success_label_btn)
            )
        # success message cell in horizontal layout
        self.success_msg_HL = QtWidgets.QHBoxLayout()
        self.success_msg_HL.setObjectName("success_msg_HL")
        self.success_msg_HL.addWidget(self.success_label)
        self.success_msg_HL.addItem(
            QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        )
        self.success_msg_HL.addWidget(self.success_label_btn)

        # folder name label
        self.folder_name_lbl = QtWidgets.QLabel(self.upload_images_page)
        self.folder_name_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.folder_name_lbl.setObjectName("folder_name_lbl")
        # folder name text field
        self.folder_name_txt = QtWidgets.QLineEdit()
        # folder name cell in horizontal layout
        self.folder_name_HL = QtWidgets.QHBoxLayout()
        self.folder_name_HL.setObjectName("folder_name_HL")
        self.folder_name_HL.addWidget(self.folder_name_lbl)
        self.folder_name_HL.addWidget(self.folder_name_txt)
        
        # upload faces page button
        self.upload_images_page_btn = QtWidgets.QPushButton(self.upload_images_page)
        self.upload_images_page_btn.clicked.connect(self.upload_faces)

        # your uploads label
        self.uploads_lbl = QtWidgets.QLabel(self.upload_images_page)
        self.uploads_lbl.setObjectName("uploads_lbl")

        # upload faces page in vertical layout
        self.upload_face_vbl.addLayout(self.success_msg_HL)
        self.upload_face_vbl.addItem(
            QtWidgets.QSpacerItem(20, 50, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        )
        self.upload_face_vbl.addLayout(self.folder_name_HL)
        self.upload_face_vbl.addWidget(self.upload_images_page_btn)
        self.upload_face_vbl.addItem(
            QtWidgets.QSpacerItem(20, 50, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        )
        self.upload_face_vbl.addWidget(self.uploads_lbl)

        self.scroll = QtWidgets.QScrollArea()
        self.widget = QtWidgets.QWidget()

        self.uploads()

        self.widget.setLayout(self.uploads_vbl)

        #Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.vbl.addLayout(self.upload_face_vbl)
        self.vbl.addWidget(self.scroll)
        self.upload_images_page.setLayout(self.vbl)

        self.stackedWidget.addWidget(self.upload_images_page)

        self.success_label.hide()
        self.success_label_btn.hide()
        
        # --------------------upload images page end--------------------

        # grid layout of content page
        self.gridLayout_7 = QtWidgets.QGridLayout(self.content_page)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.gridLayout_7.addWidget(self.stackedWidget, 0, 0, 1, 1)


        # --------------------content page widget end--------------------


        # grid layout of central widget
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        # header widget in grid layout of central widget
        self.gridLayout.addWidget(self.header, 0, 0, 1, 2)
        # sidebar widget in grid layout of central widget
        self.gridLayout.addWidget(self.sidebar, 1, 0, 1, 1)
        # content page widget in grid layour of central widget
        self.gridLayout.addWidget(self.content_page, 1, 1, 1, 1)

        # central widget in mainwindow
        MainWindow.setCentralWidget(self.centralwidget)


        self.retranslateUi(MainWindow)
        self.close_btn.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


# all about uploads

    def upload_videos(self):
        path = os.path.join(self.ROOT_DIR, 'uploaded_videos', self.upload_videos_text.text())

        try: os.mkdir(path)
        except OSError: pass
        vid_paths, _ = QFileDialog.getOpenFileNames(None, "UPLOAD VIDEOS", self.ROOT_DIR, "Videos (*.mp4)")
        i = len(os.listdir(path))

        if vid_paths:
            for vid_path in vid_paths:
                fname = self.upload_videos_text.text()+"_"+str(i)+"."+str(vid_path.split('.')[-1])
                path += '\\' + fname
                # print(path)
                # print(vid_path)
                shutil.copy(vid_path, path)

        QtWidgets.QMessageBox.information(self.upload_videos_page, 'Success', 'Videos uploaded successfully!')


        # try: os.mkdir(path)
        # except OSError: pass
        # vid_paths, _ = QFileDialog.getOpenFileNames(None, "UPLOAD VIDEOS", self.ROOT_DIR, "Videos (*.mp4)")
        # i = len(os.listdir(path))
        
        # if vid_paths:
        #     for vid_path in vid_paths:
        #         vid_cap = cv2.VideoCapture(vid_path)
        #         os.chdir(path)
        #         fname = self.upload_videos_text.text()+"_"+str(i)+"."+str(vid_path.split('.')[-1])
        #         frame_w = 1920
        #         frame_h = 1080
        #         fps = 50
        #         vid_out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_w, frame_h))
                 
        #         while(True):
        #             ret, frame = vid_cap.read()
        #             if ret == True: 
        #                 vid_out.write(frame)
        #             else:
        #                 break 
        #             vid_cap.release()
        #             vid_out.release()

        #         i+=1
        #         # self.success_label.show()
        #         # self.success_label_btn.show()
        #         # self.uploads()
        # self.upload_videos_text.clear()



    def upload_faces(self):
        # ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
        path = os.path.join(self.ROOT_DIR, 'uploads', self.folder_name_txt.text())
        try: os.mkdir(path)
        except OSError: pass
        img_paths, _ = QFileDialog.getOpenFileNames(None, "UPLOAD IMAGES", self.ROOT_DIR, "Images (*.png *.jpg *.jpeg)")
        i = len(os.listdir(path))
        
        if img_paths:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                os.chdir(path)
                fname = self.folder_name_txt.text()+"_"+str(i)+"."+str(img_path.split('.')[-1])
                # dt = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
                # fname = self.folder_name_txt.text()+"_"+dt+"."+str(img_path.split('.')[-1])
                cv2.imwrite(fname, img)
                i+=1
                self.success_label.show()
                self.success_label_btn.show()
                self.uploads()
        self.folder_name_txt.clear()

    def uploads(self):
        '''
        this method updates the list of uploaded images.
        code is buggy!
        '''
        path = os.path.join(self.ROOT_DIR, 'uploads')
        lbl = QtWidgets.QLabel(self.upload_images_page)
        # uploads list
        for i in os.listdir(path):
            inner_dir_path = os.path.join(path, i)
            if not os.path.isfile(inner_dir_path):
                for img_path in os.listdir(inner_dir_path):
                    lbl.setText(i+"\\"+img_path)
                    self.uploads_vbl.addWidget(lbl)
    
    def success_close(self, label, button):
        label.hide()
        button.hide()



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate

        # mainwindow
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        # header labels
        self.fr_label.setText(_translate("MainWindow", "FACE RECOGNIZER"))

        # sidebar buttons
        self.home_btn.setText(_translate("MainWindow", "HOME"))
        self.show_data_btn.setText(_translate("MainWindow", "SHOW DATA"))
        self.recg_face_btn.setText(_translate("MainWindow", "RECOGNIZE FACE"))
        self.sett_panel_btn.setText(_translate("MainWindow", "SETTINGS PANEL"))
        self.add_camera_btn.setText(_translate("MainWindow", "ADD CAMERA"))
        self.upload_videos_btn.setText(_translate("MainWindow", "UPLOAD VIDEOS"))
        self.upload_images_btn.setText(_translate("MainWindow", "UPLOAD IMAGES"))
        self.close_btn.setText(_translate("MainWindow", "CLOSE"))

        # content page label
        self.home_page_label.setText(_translate("MainWindow", "STREAM"))
        self.label_3.setText(_translate("MainWindow", "ADD DATA"))
        self.label_4.setText(_translate("MainWindow", "RECOGNIZE FACE"))
        
        # settings panel page labels and buttons
        self.sett_panel_label.setText(_translate("MainWindow", "SETTINGS PANEL"))
        self.detection_model_label.setText(_translate("MainWindow", "CHOOSE DETECTION MODEL"))
        self.recognition_model_label.setText(_translate("MainWindow", "CHOOSE RECOGNITION MODEL"))
        self.processors_label.setText(_translate("MainWindow", "CHOOSE PROCESSORS"))
        self.save_sett_btn.setText(_translate("MainWindow", "SAVE"))

        # upload faces page labels and buttons
        self.success_label.setText(_translate("MainWindow", "FACES UPLOADED SUCCESSFULLY"))
        self.success_label_btn.setText(_translate("MainWindow", ""))
        self.upload_images_page_btn.setText(_translate("MainWindow", "BROWSE IMAGES"))
        self.folder_name_lbl.setText(_translate("MainWindow", "ADD FOLDER NAME"))
        self.uploads_lbl.setText(_translate("MainWindow", "YOUR UPLOADS:"))


