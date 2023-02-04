import cv2 as cv
import mediapipe as mp
from joblib import Parallel, delayed
from itertools import count
from time import time

class FaceDetect:

    def __init__(self,*arg,**kwarg):
        self.queue = kwarg["queue"]
        self.username = kwarg["username"]
        self.password = kwarg["password"]
        self.IP = kwarg["IP"]
        self.start_detection()

    
    def get_bbox(self,face, img_row, img_col):
        rrb = face.location_data.relative_bounding_box
        x,y = int(img_col*rrb.xmin),int(img_row*rrb.ymin)
        width,height = int(img_col*rrb.width),int(img_row*rrb.height)

        landmarks = face.location_data.relative_keypoints
        right_eye = (int(landmarks[0].x * img_col), int(landmarks[0].y * img_row))
        left_eye = (int(landmarks[1].x * img_col), int(landmarks[1].y * img_row))
        return [(x,y),(x+width,y+height),(left_eye,right_eye)]


    def start_detection(self):

        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection = 1
        )
        drawer = mp.solutions.drawing_utils

        capture = cv.VideoCapture("video/testingvid.mp4")
        fps = capture.get(cv.CAP_PROP_FPS)
        frame_no, sec, min = 0, 0, 0
        try:
            start_time_detect = time()
            while(True):
                isTrue, frame = capture.read()
                
                if not isTrue:
                   
                    break

                frame_no += 1
                if frame_no%fps == 0:
                    sec += 1
                    frame_no = 0
                if sec%60 == 0 and sec!=0:
                    min += 1
                    sec = 0

                img_row, img_col = frame.shape[0],frame.shape[1]
                faces = face_detector.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB)).detections

                if not faces:
                    continue

                bboxes = []
                for face in faces:
                    (x,y),(w,h),(left_eye,right_eye) = self.get_bbox(face,img_row,img_col)
                    cv.rectangle(
                        frame, 
                        (x,y),
                        (w,h),
                        color = (255,0,0),
                        thickness=1
                    )
                    bboxes.append([x,y,w,h,left_eye,right_eye])

                if frame_no%6==0:
                    self.queue.put([cv.cvtColor(frame,cv.COLOR_BGR2RGB),bboxes,True,(min,sec)])

                cv.imshow("Medaipipe",frame)
                if cv.waitKey(20)&0xFF == ord("d"):
                    break
        
        except KeyboardInterrupt:
            print("Detector End")

        finally:
            self.queue.put([frame,bboxes,False,(min,sec)])
            print(f"Detector total time: Minutes: {(time()-start_time_detect)//60}, Seconds: {(time()-start_time_detect)%60}\n")
            capture.release()
            cv.destroyAllWindows()
