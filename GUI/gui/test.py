from FaceDetector.fd_mediaipipe   import MediaPipeWrapper
from FaceRecognizer.fr_template import FaceRecogTemp
import cv2 as cv

fd =  MediaPipeWrapper()
fr = FaceRecogTemp(model_name="Facenet")

def make_enc():
    img_path = "IMG_20210219_230731_Bokeh__01.jpg"
    img = cv.imread(img_path)
    face = fd.capture_faces(img)[0]
    bbox = fd.get_bbox(face,img.shape[0],img.shape[1])
    fr.create_save_encoding(
        "adnan",
        cv.cvtColor(img,cv.COLOR_BGR2RGB),
        bbox[0][0],bbox[0][1],
        bbox[1][0],bbox[1][1],
        bbox[2][0], bbox[2][1]
    )
    cv.imwrite("uploads/"+img_path,img[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]])

make_enc()