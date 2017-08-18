"""Scikit runner"""
import numpy as np
import argparse
import os
import sys
import pandas as pd

from sklearn.externals import joblib
from abc import ABC
from abc import abstractmethod
from ml_project import configparse
from pprint import pprint
from os.path import normpath


class Action(ABC):
    """Abstract Action class

    Args:
        args (Namespace): Parsed arguments
    """
    def __init__(self, args):
        self.args = args
        self.save_path = self._mk_save_folder()
        self.X, self.y = self._load_data()
        self.X_new, self.y_new = None, None
        self._X_new_set, self._y_new_set = False, False

    @abstractmethod
    def _save(self):
        pass

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _check_action(self):
        pass

    def act(self):
        self._check_action(self.args.action)
        self.model = self._load_model()
        getattr(self, self.args.action)()
        if self.args.smt_label != "debug":
            self._save()

    def _load_data(self):
        try:
            X = np.load(self.args.X)
        except FileNotFoundError:
            print("{} not found. "
                  "Please download data first.".format(self.args.X))
            exit()
        if self.args.y is not None:
            try:
                y = np.loadtxt(self.args.y)
            except FileNotFoundError:
                print("{} not found. "
                      "Please download data first.".format(self.args.y))
                exit()
        else:
            y = None
        return X, y

    def _mk_save_folder(self):
        if self.args.smt_label != "debug":
            basename = self.args.smt_label
            path = "data/"+basename+"/"
            os.mkdir(normpath(path))
            return path
        else:
            return None


class ConfigAction(Action):
    """Class to handle config file actions

    Args:
        args (Namespace): Parsed arguments
        config (dict): Parsed config file

    """
    def __init__(self, args, config):
        super(ConfigAction, self).__init__(args)
        self.config = config
        self.pprint_config()
        self.act()

    def fit(self):
        self.model.fit(self.X, self.y)

    def transform(self):
        self.X_new = self.model.transform(self.X, self.y)
        self._X_new_set = True

    def fit_transform(self):
        self.fit()
        self.transform()

    def _save(self):
        class_name = self.config["class"].__name__
        joblib.dump(self.model,
                    normpath(self.save_path+class_name+".pkl"))

        if self._X_new_set:
            path = self.save_path+"X_new.npy"
            np.save(normpath(path), self.X_new)

        if hasattr(self.config["class"], "save"):
            self.model.save(self.save_path)

    def _load_model(self):
        return self.config["class"](**self.config["params"])

    def _check_action(self, action):
        if action not in ["fit", "fit_transform"]:
            raise RuntimeError("Can only run fit or fit_transform from config,"
                               " got {}.".format(action))

    def pprint_config(self):
        print("\n=========== Config ===========")
        pprint(self.config)
        print("==============================\n")
        sys.stdout.flush()


class ModelAction(Action):
    """Class to model actions

    Args:
        args (Namespace): Parsed arguments
    """
    def __init__(self, args):
        super(ModelAction, self).__init__(args)
        self._check_action(args.action)
        self.act()

    def transform(self):
        self.X_new = self.model.transform(self.X, self.y)
        self._X_new_set = True

    def predict(self):
        self.y_new = self.model.predict(self.X)
        self._y_new_set = True

    def _save(self):
        if self._X_new_set:
            np.save(normpath(self.save_path+"X_new.npy"), self.X_new)
        if self._y_new_set:
            df = pd.DataFrame({"Prediction": self.y_new})
            df.index += 1
            df.index.name = "ID"
            df.to_csv(normpath(self.save_path+"y_"+self.args.smt_label+".csv"))

    def _load_model(self):
        return joblib.load(self.args.model)

    def _check_action(self, action):
        if action not in ["transform", "predict"]:
            raise RuntimeError("Can only run transform or predict from model,"
                               " got {}.".format(action))


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Scikit runner.")

    arg_parser.add_argument("-C", "--config", help="config file")
    arg_parser.add_argument("-M", "--model", help="model file")

    arg_parser.add_argument("-X", help="Input data", required=True)
    arg_parser.add_argument("-y", help="Input labels")

    arg_parser.add_argument("-a", "--action", choices=["transform", "predict",
                            "fit", "fit_transform"], help="Action to perform.",
                            required=True)

    arg_parser.add_argument("smt_label", nargs="?", default="debug")

    args = arg_parser.parse_args()

    if args.config is None:
        ModelAction(args)
    else:
        config_parser = configparse.ConfigParser()
        config = config_parser.parse_config(args.config)
        ConfigAction(args, config)
