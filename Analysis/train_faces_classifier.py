from save_load_encoding import load_encoding_json
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, LabelEncoder
from joblib import dump, load
import numpy as np
import sys


def train_svm(file_path,classifier_name="facenet"):
    names, features_ls = load_encoding_json(file_path)
    in_encoder = Normalizer(norm="l2")
    features_ls = in_encoder.transform(features_ls)
    out_encoder = LabelEncoder()
    out_encoder.fit(names)
    output = out_encoder.transform(names)
    model = LinearSVC()
    model.fit(features_ls,output)
    dump(model,classifier_name+"_svm_clf.joblib")
    dump(out_encoder,classifier_name+"_svm_enc_out.joblib")

def train_knn(file_path,classifier_name="facenet"):
    names, features_ls = load_encoding_json(file_path)
    out_encoder = LabelEncoder()
    output = out_encoder.fit_transform(names)
    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(features_ls,output)
    dump(model,classifier_name+"_knn_clf.joblib")
    dump(out_encoder,classifier_name+"_knn_enc_out.joblib")

if __name__ == '__main__':
    if len(sys.argv) <=1:
        raise IndexError("Not Enough Arguments")

    if sys.argv[1] == "svm":
        train_svm("feature_encoding.json")
    elif sys.argv[1] == "knn":
        train_knn("feature_encoding.json")
    else:
        raise ValueError(f"No classifier of type: {sys.argv[1]}")