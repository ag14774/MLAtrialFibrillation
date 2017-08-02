import numpy as np
import argparse
import configparse
import os
import datetime
import pandas as pd

from sklearn.externals import joblib
from abc import ABC
from abc import abstractmethod


class Action(ABC):
    """docstring for Action"""
    def __init__(self, args): 
        self.args = args     
        self.X, self.y = self.load_data()
        self.X_new, self.y_new = None, None
        self._X_new_set, self._y_new_set = False, False
        self.save_path = self.mk_save_folder()

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    def load_data(self):
        X = np.load(self.args.X)
        if self.args.y != None:
            y = np.loadtxt(self.args.y)
        else:
            y = None
        return X, y

    def mk_save_folder(self):
        basename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.name:
            name = basename+"-"+args.name
        else:
            name = basename
        path = "archive/"+name+"/"
        os.mkdir(path)
        return path


class ConfigAction(Action):
    """docstring for ConfigAction"""
    def __init__(self, args, config):
        super(ConfigAction, self).__init__(args)
        self.check_config(config)
        self.config = config
        self.model = self.load_model()       
        getattr(self, config["action"])()
        self.save()

    def fit(self):
        self.model.fit(self.X, self.y)

    def _transform(self):        
        self.X_new = self.model.transform(self.X, self.y)
        self._X_new_set = True

    def fit_transform(self):
        self.fit()
        self._transform()

    def save(self):
        name = self.config["class"].__name__
        joblib.dump(self.model, self.save_path+name+".pkl")
       
        if self._X_new_set:
            np.save(self.save_path+"X_new.npy", self.X_new)  

    def load_model(self):
        return self.config["class"](**self.config["params"])

    def check_config(self, config):
        if config["action"] not in ["fit", "fit_transform"]:
            raise Error("Can only run fit or fit_transform from config, got {}."
                        .format(config["action"]))

        if not config["class"]:
            raise Error("Model class not specified in config file.")

class ModelAction(Action):
    """docstring for ModelAction"""
    def __init__(self, args):
        super(ModelAction, self).__init__(args)
        self.model = self.load_model()       
        getattr(self, args.action)()
        self.save()

    def transform(self):        
        self.X_new = self.model.transform(self.X, self.y)
        self._X_new_set = True

    def predict(self):
        self.y_new = self.model.predict(self.X)
        self._y_new_set = True

    def save(self):
        if self._X_new_set:
            np.save(self.save_path+"X_new.npy", self.X_new) 
        if self._y_new_set:
            df = pd.DataFrame({"Prediction": self.y_new})
            df.index += 1
            df.index.name = "ID"
            df.to_csv(self.save_path+"y_new.csv")

    def load_model(self):
        return joblib.load(self.args.model)

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Scikit runner.")

    subparsers = arg_parser.add_subparsers()
    from_config = subparsers.add_parser("config", help="Run from config file.")
    from_model = subparsers.add_parser("model", help="Run from stored model")

    from_config.add_argument("config", help="Path to config file.")
    from_config.add_argument("-X", help="Input data", default=None, required=True)
    from_config.add_argument("-y", help="Input labels", default=None)
    from_config.add_argument("-N", "--name", help="Output folder name")
    
    from_model.add_argument("model", help="Path to fitted model.")
    from_model.add_argument("-a", "--action", choices=["transform", "predict"],
                            help="Action to perform.", required=True)
    from_model.add_argument("-X", help="Input data", default=None, required=True)
    from_model.add_argument("-y", help="Input labels", default=None)
    from_model.add_argument("-N", "--name", help="Output folder name")

    
    args = arg_parser.parse_args()

    try:
        config_parser = configparse.ConfigParser()
        config = config_parser.parse_config(args.config)
        ConfigAction(args, config)
    except AttributeError:
        ModelAction(args)
