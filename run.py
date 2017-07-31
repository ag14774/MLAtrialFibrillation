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
    def __init__(self, args, config=None):
        self.config = config
        self.args = args

        self.model = self.load_model()       
        self.X, self.y = self.load_data()
        self.X_new, self.y_new = None, None

        self._X_new_set = False
        self._y_new_set = False

        if self.config:
            getattr(self, config["action"])()
        else:
            getattr(self, args.action)()

        self.save()

    def fit(self):
        self.model.fit(self.X, self.y)

    def transform(self):        
        self.X_new = self.model.transform(self.X, self.y)
        self._X_new_set = True

    def fit_transform(self):
        self.fit()
        self.transform()

    def predict(self):
        self.y_new = self.model.predict(self.X)
        self._y_new_set = True

    def save(self):
        if self.config["outputs"] != None:

            time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = "archive/"+time+"/"
            os.mkdir(path)

            for output in self.config["outputs"]:

                if output["type"] == "data":
                    if self._X_new_set:
                        np.save(path+"X_new.npy", self.X_new)
                    if self._y_new_set:
                        np.save(path+"y_new.npy", self.y_new)

                if output["type"] == "model":
                    name = self.config["class"].__name__
                    joblib.dump(self.model, path+name+".pkl")

    def load_data(self):
        X = np.load(self.args.X)

        if self.args.y != None:
            y = np.load(self.args.y)
        else:
            y = None

        return X, y

    def load_model(self):

        if self.args.model:
            return joblib.load(self.args.model)
        elif self.config["class"]:
            return self.config["class"](**self.config["params"])
        else:
            raise RuntimeError("Model was not be specified.")

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Scikit runner.")
    arg_parser.add_argument("-c", "--config", help="Path to config file")
    arg_parser.add_argument("-X", help="Input data", default=None)
    arg_parser.add_argument("-y", help="Input labels", default=None)
    arg_parser.add_argument("-m", "--model", help="Fitted model", default=None)
    arg_parser.add_argument("-a", "--action", help="Action to perform")
    args = arg_parser.parse_args()

    if args.config:
        config_parser = configparse.ConfigParser()
        config = config_parser.parse_config(args.config)
        Action(args, config)
    else:
        Action(args)
