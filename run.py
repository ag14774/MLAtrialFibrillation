import numpy as np
import argparse
import configparse
import tarfile
from io import BytesIO
import os
import datetime
from sklearn.externals import joblib

#y = np.loadtxt("data/y_1.csv")
"""
def load_from_record(record):
    recordID = record.split("/")[1].split(".")[0]
    tar = tarfile.open(record)
    file = tar.extractfile(recordID+"/X_test_rnd_sel.npy")
    byte_obj = BytesIO(file.read())
    return np.load(byte_obj)
"""
class Action(object):
    """docstring for Action"""

    def __init__(self, config, args):
        self.config = config
        self.args = args

        self.model = self.load_model()       
        self.X, self.y = self.load_data()
        self.X_new, self.y_new = None, None

        getattr(self, config["action"])()

        self.save()

    def fit(self):

        self.model.fit(self.X, self.y)

        for output in self.config["outputs"]:
            if output["type"] == "model":
                self.save_model(self.model)

    def transform(self):
        
        X_new, y_new = self.model.transform(self.X, self.y)

        for output in self.config["outputs"]:
            if output["type"] == "data":
                self.save_data(X_new, y_new)

    def predict(self):
        pass

    def fit_transform(self):
        pass

    def save(self):
        if self.config["outputs"] not None:

            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            path = "archive/"+time+"/"
            os.mkdir("archive/"+time)

            for output in self.config["outputs"]:

                if output["type"] == "data":
                    if self.X_new:



    def save_model(self, model):
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.mkdir("archive/"+time)
        name = self.config["class"].__name__
        path = "archive/"+time+"/"+name+".pkl"
        joblib.dump(model, path)

    def save_data(self, X=None, y=None):
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.mkdir("archive/"+time)
        name = self.config["class"].__name__
        path = "archive/"+time+"/"+name+".pkl"
        joblib.dump(model, path)

    def load_data(self):
        X = np.load(self.args.X)

        if self.args.y:
            y = np.load(self.args.y)
        else:
            y = None

        return X, y

    def load_model(self):
        return self.config["class"](**self.config["params"])

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Scikit runner.")
    arg_parser.add_argument("configfile", help="Path to config file")
    arg_parser.add_argument("-X", help="Input data", default=None)
    arg_parser.add_argument("-y", help="Input labels", default=None)
    args = arg_parser.parse_args()

    config_parser = configparse.ConfigParser()
    config = config_parser.parse_config(args.configfile)

    Action(config, args)

    #getattr(Action, config["action"])(config, args)

    """
    model = config["class"](**config["params"])
    action = getattr(model, config["action"])

    X = np.load(args.X)#load_from_record(args.X)
    X_new = action(X)

    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.mkdir("archive/"+time)
    for output in config["outputs"]:
        if output["type"] == "data":
            path = "archive/"+time+"/"+"X.npy"
            np.save(path, X_new)
        elif output["type"] == "model":
            name = config["class"].__name__
            path = "archive/"+time+"/"+name+".pkl"
            joblib.dump(model, path)
    """
