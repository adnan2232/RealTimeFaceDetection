import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream

from gui_ui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.menu_btn.clicked[bool].connect(self.changeState)

        self.ui.stackedWidget.setCurrentIndex(0)

        
        self.ui.sidebar.hide()
    

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