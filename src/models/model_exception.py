class ModelException(Exception):
    """
    Exception class for the model.
    """

    def __init__(self, message: str) -> None:
        self.message = message
