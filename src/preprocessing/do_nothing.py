# this will be a subcalss of the Preprocessor class
# it should just return the data as is

from src.preprocessing.pp import PP


class DoNothing(PP):
    def __init__(self):
        pass

    def preprocess(self, data):
        return data
