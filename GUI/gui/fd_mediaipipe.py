import mediapipe as mp

class MediaPipeWrapper:

    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection = 1
        )

    def get_bbox(self,face,img_row,img_col):
        rrb = face.location_data.relative_bounding_box
        x,y = int(img_col*rrb.xmin),int(img_row*rrb.ymin)
        width,height = int(img_col*rrb.width),int(img_row*rrb.height)

        landmarks = face.location_data.relative_keypoints
        right_eye = (int(landmarks[0].x * img_col), int(landmarks[0].y * img_row))
        left_eye = (int(landmarks[1].x * img_col), int(landmarks[1].y * img_row))
        return [(x,y),(x+width,y+height),(left_eye,right_eye)]


    def capture_faces(self,frame):
        return self.face_detector.process(frame).detections