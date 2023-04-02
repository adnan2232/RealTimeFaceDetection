from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from FaceRecognizer.fr_template import FaceRecogTemp
from joblib import Parallel, delayed
from itertools import count
from datetime import date
import numpy as np
import cv2 as cv
import csv
import os


class FaceRecognition(QThread):
    def __init__(self, *arg, **kwarg):

        super(FaceRecognition, self).__init__()

        self.queue = kwarg["queue"]
        self.model_name = kwarg["model_name"]
        self.model = FaceRecogTemp(self.model_name)
        self.curr_date = date.today()

        if not os.path.isdir("face_seen"):
            os.mkdir("face_seen")

        self.seen_file = open(f"face_seen/{self.curr_date.isoformat()}.csv", "a")
        self.writer = csv.writer(self.seen_file)
        self.writerTimer = QTimer(self)
        self.writerTimer.timeout.connect(self.writeCSV)
        self.writerTimer.start(50000)

        self.flushTimer = QTimer(self)
        self.writerTimer.timeout.connect(self.seen_file.flush)
        self.writerTimer.start(60000)

        self.seen = {}

        self._run_flag= True

    def writeCSV(self):
        if self.seen_file.closed:
            self.seen_file = open(f"face_seen/{self.curr_date.isoformat()}.csv", "a")
            self.writer = csv.writer(self.seen_file)
        
        for key in self.seen:
            
            self.writer.writerow(
                [key,self.seen[key][0],self.seen[key][1]]
            )

        self.seen = {}
           
    def close_file(self):
        self.seen_file.flush()
        self.seen_file.close()

    def buffer_seen(self,res,f_time):
        if res:
            for name in res[0]:
                if name in self.seen:
                    self.seen[name][1] = f_time
                else:
                    self.seen[name] = [f_time,f_time]

    def run(self):
        try:

            while not self.isInterruptionRequested():

                if self.curr_date < date.today():
                    self.close_file()
                    self.curr_date = date.today()
                    self.seen_file = open(
                        f"face_seen/{self.curr_date.isoformat()}.csv", "a"
                    )
               
                if self.queue.empty():
                    continue
                
                frame, bboxes, f_time = self.queue.get()
                emb_v = self.model.create_encodings(frame, bboxes)
                
                res = self.model.predicts([emb for emb in emb_v if emb!=[]])
                self.buffer_seen(res,f_time)
            
                del frame
                del bboxes

        except KeyboardInterrupt:
            print("recognition stop")

        finally:
            self.stop(empty_q=True)

    def stop(self, empty_q: bool = True):
     
        if empty_q:
            print("yes")
            while not self.queue.empty():
                
                frame, bboxes, f_time = self.queue.get()
                emb_v = self.model.create_encodings(frame, bboxes)
                res = self.model.predicts([emb for emb in emb_v if emb!=[]])
                self.buffer_seen(res,f_time)

                del frame
                del bboxes

            self.writeCSV()
            self.close_file()

        self._run_flag = False
     
        
