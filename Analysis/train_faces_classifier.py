from save_load_encoding import load_encoding_json
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer, LabelEncoder
from joblib import dump, load
import numpy as np


def train_svm(file_path,classifier_name="facenet"):
    names, features_ls = load_encoding_json(file_path)
    in_encoder = Normalizer(norm="l2")
    features_ls = in_encoder.transform(features_ls)
    out_encoder = LabelEncoder()
    out_encoder.fit(names)
    output = out_encoder.transform(names)
    model = LinearSVC()
    model.fit(features_ls,output)
    if classifier_name == "facenet":
        dump(model,"facenet_clf.joblib")
        dump(out_encoder,"facenet_enc_out.joblib")

if __name__ == '__main__':

    train_svm("feature_encoding.json")