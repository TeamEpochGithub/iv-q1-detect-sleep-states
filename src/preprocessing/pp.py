

class PP:
    def __init__(self):
        self.use_pandas = True

    def preprocess(self, data):
        raise NotImplementedError

    def run(self, data):
        return self.preprocess(data)


class PPException(Exception):
    pass
