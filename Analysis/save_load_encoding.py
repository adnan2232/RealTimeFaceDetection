import json
import numpy as np

def save_encoding_json(features_ls,names,file_name):

    with open(file_name,"w+") as fc:
        try:
            encoded_faces = json.load(fc)
        except:
            encoded_faces = []
        finally:
            encoded_faces.extend([{"name":name,"features":features} for features,name in zip(features_ls,names)])
            json.dump(
                encoded_faces,
                fc
            )

def load_encoding_json(file_name):
    names = []
    features = []

    try:
        with open(file_name,"r") as fc:
            encoded_faces = json.load(fc)

            for info in encoded_faces:
                names.append(info["name"])
                features.append(np.array(info["features"]))
    except:
        pass
    finally:
        return names, features