# Before you dive into it
This project is a prototype few bugs maybe crawling somewhere but its a good place to start if you're looking for some integration of traiditional, deep learning and machine learning algorithms based real-time application

# Real-Time Face Detection Recognition
This application is created for Real-time face detection and recognition integrating with CCTV (rtsp protocol), cloud repo or any stored-file in pc. It leverages the power of multiple cores

# Analysis
Analysis content some research testing files and research based on Multi-processing do not confuse it with GUI application which is based on concurrent processing, you can run analysis from main.py but before running store your encoding using create_features.py

# GUI/ PyQt based cross-platform application
Create seperate environment to use this application and use its own [requirements.txt](GUI/requirements.txt) you can directly run the app by running python3/python main.py from [gui folder](GUI/gui). You can made your own changes to this application. If use want to test your own face recognizer model just create a [modularized file](GUI/gui/FaceRecognizer) and in fr_template replace [DeepFace.represnt](GUI/gui/FaceRecognizer/fr_template) in create_encoding method with your own model that create embeddings. You can also seperate KNN module from fr_template or use SVM as you wish. For face detector just create Wrapper for it checkout [Mediapipe and MTCNN wrapper](GUI/gui/FaceDetector). You can also use some different library for receiving live stream other than opencv because opencv wasn't created for receiving livestream but it still can be use for prototypes (:. If you're receiving stream for high fps camera you can also increase frame dropping limited thresholds. [GUI](GUI/gui/gui_ui.py) is created using pyqt-tool. I recommend you to modularize gui and seperate some methods from [main.py](GUI/gui/main.py)

*Note: TinyDB is not Multi-threading safe so you cannot create a new object for it in different file until a thread accessing it is running, Stop the thread before accessing it. (You can replace TinyDB with some thread safe database (sqlitedb) it can also increase the performance because stopping and restarting thread often degrade the performance)*
