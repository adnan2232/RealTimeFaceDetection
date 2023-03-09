from PyQt5.QtCore import QThread, pyqtSignal
from FaceRecognizer.fr_template import FaceRecogTemp
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
       
        self._run_flag= True

    def close_file(self):
        self.seen_file.flush()
        self.seen_file.close()


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
                
                res = self.model.predicts(emb_v)
          
                if res:
                    self.writer.writerows(
                        zip(*res,[f_time]*len(res))
                    )

        except KeyboardInterrupt:
            print("recognition stop")

        finally:
            while not self.queue.empty():
                frame, bboxes, f_time = self.queue.get()
                emb_v = self.model.create_encodings(frame, bboxes)
                res = self.model.predicts(emb_v)
                
                if res:
                    self.writer.writerows(
                        zip(*res,[f_time]*len(res))
                    )

            self.close_file()

    def stop(self):

        while not self.queue.empty():
            frame, bboxes, f_time = self.queue.get()
            emb_v = self.model.create_encodings(frame, bboxes)
            res = self.model.predicts(emb_v)
            if res:
                self.writer.writerows(
                    zip(*res,[f_time]*len(res))
                )

        self.close_file()
        self._run_flag = False
