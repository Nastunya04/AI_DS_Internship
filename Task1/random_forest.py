"""
RandomForestClassifierModel

This module contains the implementation of the Random Forest classifier for MNIST classification.
The model supports optional hyperparameter tuning using RandomizedSearchCV and provides
evaluation metrics.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from interface import MnistClassifierInterface


class RandomForestClassifierModel(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST digit classification.
    """

    def __init__(self, n_estimators=50, random_state=42, tune=False):
        """
        Initializes the Random Forest classifier.

        Args:
            n_estimators: number of trees in the forest.
            random_state: random seed for reproducibility.
            tune: whether to perform hyperparameter tuning.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.tune = tune
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, \
        random_state=self.random_state)

    def hyperparameter_tuning(self, X_train, y_train):
        """
        Performs hyperparameter tuning using StratifiedKFold & RandomizedSearchCV.

        Args:
            X_train: feature matrix for training.
            y_train: corresponding training labels.

        Returns:
            None
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'max_features': ['sqrt', 'log2', None]
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        search = RandomizedSearchCV(
            self.model, param_distributions=param_grid, n_iter=10,
            scoring='accuracy', cv=skf, n_jobs=-1, verbose=2, random_state=self.random_state
        )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        print(f"Best Hyperparameters: {search.best_params_}")

    def train(self, X_train, y_train):
        """
        Trains the Random Forest model.

        Args:
            X_train: training feature matrix.
            y_train: corresponding labels for training data.

        Returns:
            None
        """
        if self.tune:
            print("Running Hyperparameter Tuning...")
            self.hyperparameter_tuning(X_train, y_train)

        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts labels using the trained Random Forest model.

        Args:
            X_test: feature matrix for testing.

        Returns:
            predicted class labels.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model's performance using accuracy, precision, recall, and F1-score.

        Args:
            X_test: feature matrix for testing.
            y_test: true labels for testing data.

        Returns:
            accuracy score of the model.
        """
        y_pred = self.predict(X_test)
        print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
        return accuracy_score(y_test, y_pred)

    def save_model(self, filename="random_forest_model.pkl"):
        """
        Saves the trained Random Forest model to a file.

        Args:
            filename: path to save the model.

        Returns:
            None
        """
        joblib.dump(self.model, filename)
        print(f"Model saved as {filename}")

    def load_model(self, filename="random_forest_model.pkl"):
        """
        Loads a trained Random Forest model from a file.

        Args:
            filename: path to the saved model.

        Returns:
            None
        """
        try:
            self.model = joblib.load(filename)
            print(f"Loaded model from {filename}")
        except FileNotFoundError:
            print(f"No saved model found at {filename}. Try training a new one.")
