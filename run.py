"""Scikit runner"""
import numpy as np
import argparse
import os
import datetime
import pandas as pd

from sklearn.externals import joblib
from abc import ABC
from abc import abstractmethod
from ml_project import configparse

class Action(ABC):
    """Abstract Action class
    
    Args:
        args (Namespace): Parsed arguments
    """
    def __init__(self, args): 
        self.args = args     
        self.X, self.y = self._load_data()
        self.X_new, self.y_new = None, None
        self._X_new_set, self._y_new_set = False, False
        self.save_path = self._mk_save_folder()

    @abstractmethod
    def _save(self):
        pass

    @abstractmethod
    def _load_model(self):
        pass

    def _load_data(self):
        X = np.load(self.args.X)
        if self.args.y != None:
            y = np.loadtxt(self.args.y)
        else:
            y = None
        return X, y

    def _mk_save_folder(self):
        #basename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        basename = self.args.smt_label
        path = "data/"+basename+"/"
        return path


class ConfigAction(Action):
    """Class to handle config file actions

    Args:
        args (Namespace): Parsed arguments
        config (dict): Parsed config file

    """
    def __init__(self, args, config):
        super(ConfigAction, self).__init__(args)
        self._check_config(config)
        self.config = config
        self.model = self._load_model()       
        getattr(self, config["action"])()
        self._save()

    def fit(self):
        self.model.fit(self.X, self.y)

    def transform(self):        
        self.X_new = self.model.transform(self.X, self.y)
        self._X_new_set = True

    def fit_transform(self):
        self.fit()
        self.transform()

    def _save(self):
        name = self.config["class"].__name__
        joblib.dump(self.model, self.save_path+name+".pkl")
       
        if self._X_new_set:
            path = self.save_path+"X_new.npy"
            np.save(path, self.X_new)

    def _load_model(self):
        return self.config["class"](**self.config["params"])

    def _check_config(self, config):
        if config["action"] not in ["fit", "fit_transform"]:
            raise Error("Can only run fit or fit_transform from config, got {}."
                        .format(config["action"]))

        if not config["class"]:
            raise Error("Model class not specified in config file.")

class ModelAction(Action):
    """Class to model actions

    Args:
        args (Namespace): Parsed arguments
    """
    def __init__(self, args):
        super(ModelAction, self).__init__(args)
        self.model = self._load_model()       
        getattr(self, args.action)()
        self._save()

    def transform(self):        
        self.X_new = self.model.transform(self.X, self.y)
        self._X_new_set = True

    def predict(self):
        self.y_new = self.model.predict(self.X)
        self._y_new_set = True

    def _save(self):
        if self._X_new_set:
            np.save(self.save_path+"X_new.npy", self.X_new) 
        if self._y_new_set:
            df = pd.DataFrame({"Prediction": self.y_new})
            df.index += 1
            df.index.name = "ID"
            df.to_csv(self.save_path+"y_new.csv")

    def _load_model(self):
        return joblib.load(self.args.model)

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Scikit runner.")

    subparsers = arg_parser.add_subparsers()
    from_config = subparsers.add_parser("config", help="Run from config file.")
    from_model = subparsers.add_parser("model", help="Run from stored model")

    from_config.add_argument("config", help="Path to config file.")
    from_config.add_argument("smt_label")
    from_config.add_argument("-X", help="Input data", default=None, required=True)
    from_config.add_argument("-y", help="Input labels", default=None)
    from_config.add_argument("-N", "--name", help="Output folder name")
    
    from_model.add_argument("model", help="Path to fitted model.")
    from_model.add_argument("smt_label")
    from_model.add_argument("-a", "--action", choices=["transform", "predict"],
                            help="Action to perform.", required=True)
    from_model.add_argument("-X", help="Input data", default=None, required=True)
    from_model.add_argument("-y", help="Input labels", default=None)
    from_model.add_argument("-N", "--name", help="Output folder name")

    
    args = arg_parser.parse_args()

    try:
        args.config
    except AttributeError:
        ModelAction(args)
    else:
        config_parser = configparse.ConfigParser()
        config = config_parser.parse_config(args.config)
        ConfigAction(args, config)
