from sklearn.pipeline import Pipeline


class Pipeline(Pipeline):
    """docstring for Pipeline"""

    def __init__(self, class_list, save_path=None):
        self.class_list = class_list
        self.steps = self.load_steps(class_list)
        super(Pipeline, self).__init__(self.steps)
        self.set_save_path(save_path)

    def load_steps(self, class_list):
        steps = []
        for dict_ in class_list:
            name = dict_["class"].__name__
            if "params" in dict_:
                params = dict_["params"]
                steps.append((name, dict_["class"](**params)))
            else:
                steps.append((name, dict_["class"]()))
        return steps

    def set_save_path(self, save_path):
        self.save_path = save_path
        for dict_ in self.class_list:
            if hasattr(dict_["class"], "set_save_path"):
                param = {dict_["class"].__name__+"__save_path": save_path}
                self.set_params(**param)
