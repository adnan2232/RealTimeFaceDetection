from FaceDetectorTest import FaceDetect
from FaceRecogTest import FaceRecognition
from multiprocessing import Process, Queue
from time import time

if __name__ == "__main__":
    try:
        
        queue = Queue(maxsize=1000)
        
        detector_input = {
            "username":"aa2232786",
            "password": "aa2232786",
            "IP":"192.168.1.105",
            "queue":queue
        }
        classifier = "knn"
        recog_input = {
            "queue":queue,
            "clf_file":f"facenet_{classifier}_clf.joblib",
            "name_enc_file":f"facenet_{classifier}_enc_out.joblib",
            "encoding_path":"feature_encoding.json",
            "classifier":classifier
        }
        
        start_time_main = time()
        detector_process = Process(target=FaceDetect,kwargs=detector_input)
        recog_process = Process(target=FaceRecognition,kwargs=recog_input)

        detector_process.start()
        recog_process.start()
        detector_process.join()
        recog_process.join()
    except KeyboardInterrupt:
        pass

    finally:
        print(f"Main total time: {(time()-start_time_main)//60}, Seconds: {(time()-start_time_main)%60}\n")
        
        


