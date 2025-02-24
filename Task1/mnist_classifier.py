import numpy as np
from random_forest import RandomForestClassifierModel
from feedforward_nn import FeedForwardNNModel
from cnn import CNNModel


class MnistClassifier:
    """
    A unified interface for MNIST classification using Random Forest, 
    Feed-Forward Neural Network, or Convolutional Neural Network.
    """

    def __init__(self, algorithm, batch_size=32, tune=False):
        """
        Initializes the classifier based on the selected algorithm.

        Args:
            algorithm: The type of classifier ('rf', 'nn', or 'cnn').
            kwargs: Additional parameters to configure the model.
        """
        self.algorithm = algorithm.lower()
        self.batch_size = batch_size
        self.tune = tune
        if self.algorithm == "rf":
            self.model = RandomForestClassifierModel(tune=self.tune)
        elif self.algorithm == "nn":
            self.model = FeedForwardNNModel(batch_size=self.batch_size, tune=self.tune)
        elif self.algorithm == "cnn":
            self.model = CNNModel(batch_size=self.batch_size, tune=self.tune)
        else:
            raise ValueError("Invalid algorithm. Choose from: 'rf', 'nn', 'cnn'.")

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the selected model.

        Args:
            X_train: Training feature matrix.
            y_train: Corresponding training labels.
            kwargs: Additional parameters for training (e.g., epochs, batch_size).
        """
        self.model.train(X_train, y_train, **kwargs)

    def predict(self, X_test) -> np.ndarray:
        """
        Predicts labels using the selected model.

        Args:
            X_test: Feature matrix for testing.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model performance.

        Args:
            X_test: Feature matrix for testing.
            y_test: True labels for testing data.

        Returns:
            Accuracy score.
        """
        return self.model.evaluate(X_test, y_test)

    def save_model(self, filename: str):
        """
        Saves the trained model.

        Args:
            filename: The file path to save the model.
        """
        self.model.save_model(filename)

    def load_model(self, filename: str):
        """
        Loads a previously trained model.

        Args:
            filename: The file path of the saved model.
        """
        self.model.load_model(filename)
