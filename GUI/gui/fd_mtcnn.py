from mtcnn_cv2 import MTCNN

class MTCNNWrapper:

    def __init__(self):
        self.face_detector = MTCNN()
        
    def get_bbox(self,face,img_row,img_col):
        x,y,width,height = face['box']
        left_eye, right_eye = face["keypoints"]["left_eye"], face["keypoints"]["right_eye"]
        return [(x,y),(x+width,y+height),(left_eye,right_eye)]


    def capture_faces(self,frame):
        return self.face_detector.detect_faces(frame)