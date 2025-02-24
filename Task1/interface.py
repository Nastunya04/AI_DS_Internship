"""
MnistClassifierInterface

This is an abstract base class (ABC) that defines the interface for all MNIST classifiers.
Each classifier implements the `train` and `predict` methods.
"""

from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    Abstract base class for MNIST classifiers.

    All classifiers inherit from this interface and implement the required methods.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model using the provided training data.

        Args:
            X_train: feature matrix for training.
            y_train: corresponding labels for training data.

        Returns:
            None
        """


    @abstractmethod
    def predict(self, X_test):
        """
        Predicts labels for the given test data.

        Args:
            X_test: feature matrix for testing.

        Returns:
            array: predicted class labels.
        """
