# This is a sample model file. You can use this as a template for your own models.
# The model file should contain a class that inherits from the Model class.
from src.models.model import Model


class ExampleModel(Model):
    # Create different training function
    def train(self, data):
        # Train function
        print("Training model")
