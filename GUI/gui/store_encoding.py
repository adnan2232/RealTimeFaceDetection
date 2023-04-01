from FaceDetector.fd_mediaipipe   import MediaPipeWrapper
from FaceRecognizer.fr_template import FaceRecogTemp
import cv2 as cv


def store_video_enc(vid_path,name):
    fd =  MediaPipeWrapper()
    models = [FaceRecogTemp(model_name) for model_name in FaceRecogTemp.models]
    cap = cv.VideoCapture(vid_path)
    frame_no = 0
    fps = int(cap.get(cv.CAP_PROP_FPS))
    while(True):
        isFrame, frame = cap.read()
        if not isFrame:
           break
        frame_no += 1
        if frame_no%fps:
            continue
  
        faces = fd.capture_faces(frame)
        if faces:
            bbox = fd.get_bbox(faces[0],frame.shape[0],frame.shape[1])
            for model in models:
               
                model.create_save_encoding(
                    name,
                    cv.cvtColor(frame,cv.COLOR_BGR2RGB),
                    bbox[0][0],bbox[0][1],
                    bbox[1][0],bbox[1][1],
                    bbox[2][0], bbox[2][1]
                )
    cap.release()


def store_image_enc(img_paths,name):
    fd =  MediaPipeWrapper()
    models = [FaceRecogTemp(model_name) for model_name in FaceRecogTemp.models]

    for img_path in img_paths:
        img = cv.imread(img_path)
        faces = fd.capture_faces(img)

        if faces:
            bbox = fd.get_bbox(faces[0],img.shape[0],img.shape[1])

            for model in models:
                model.create_save_encoding(
                    name,
                    cv.cvtColor(img,cv.COLOR_BGR2RGB),
                    bbox[0][0],bbox[0][1],
                    bbox[1][0],bbox[1][1],
                    bbox[2][0], bbox[2][1]
                )
