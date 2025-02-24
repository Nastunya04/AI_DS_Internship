"""
CNNModel

This module implements a Convolutional Neural Network (CNN) for MNIST digit classification.
It includes convolutional layers, batch normalization, max pooling, dropout for regularization,
and supports training, prediction, evaluation, cross-validation, and hyperparameter tuning.
"""
from itertools import product
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
import interface


class CNNModel(interface.MnistClassifierInterface):
    """
    Convolutional Neural Network (CNN) for MNIST classification.
    """

    def __init__(self, input_shape=(28, 28, 1), num_classes=10, dropout_rate=0.2,
                 optimizer='adam', batch_size=8, tune=False):
        """
        Initializes the CNN model.

        Args:
            input_shape: shape of the input image.
            num_classes: number of output classes.
            dropout_rate: dropout probability for regularization.
            optimizer: optimization algorithm.
            batch_size: number of samples per batch.
            tune: whether to perform hyperparameter tuning before training.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.tune = tune
        self.model = self.build_model(optimizer)

    def build_model(self, optimizer):
        """
        Builds and compiles the CNN model.

        Args:
            optimizer: optimizer for training.

        Returns:
            compiled Keras model.
        """
        model = keras.Sequential([
            keras.layers.Input(shape=self.input_shape),

            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def hyperparameter_tuning(self, X_train, y_train, n_splits=5):
        """
        Performs manual hyperparameter tuning using Stratified K-Fold cross-validation.

        Args:
            X_train: training feature matrix.
            y_train: corresponding training labels.
            n_splits: number of cross-validation folds.

        Returns:
            None
        """

        param_grid = {
            'optimizer': ['adam'],
            'batch_size': [16, 32, 64],
            'dropout_rate': [0.2, 0.3, 0.4]
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        best_score = 0
        best_params = {}

        for optimizer, batch_size, dropout_rate in product(param_grid['optimizer'],
                                                            param_grid['batch_size'],
                                                            param_grid['dropout_rate']):
            fold_scores = []
            print(f"\nTesting: optimizer={optimizer}, batch_size={batch_size}, \
dropout_rate={dropout_rate}")
            self.dropout_rate = dropout_rate
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                print(f"  Fold {fold+1}/{n_splits}...")

                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                model = self.build_model(optimizer)
                early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', \
                patience=2, restore_best_weights=True)

                model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=batch_size,
                        validation_data=(X_val_fold, y_val_fold), callbacks=[early_stop], verbose=0)

                val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
                fold_scores.append(val_accuracy)

            avg_score = np.mean(fold_scores)
            print(f"  -> Average Validation Accuracy: {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_params = {'optimizer': optimizer, 'batch_size': batch_size, \
                'dropout_rate': dropout_rate}

        self.optimizer = best_params['optimizer']
        self.batch_size = best_params['batch_size']
        self.dropout_rate = best_params['dropout_rate']
        self.model = self.build_model(self.optimizer)

        print(f"\nBest Hyperparameters: {best_params}, Best Accuracy: {best_score:.4f}")

    def train(self, X_train, y_train, epochs=100, early_stopping=True):
        """
        Trains the CNN model with automatic Early Stopping.

        Args:
            X_train: training feature matrix.
            y_train: corresponding training labels.
            epochs: number of training iterations.
            early_stopping: whether to enable early stopping (default: True).

        Returns:
            None
        """
        if self.tune:
            print("Running Hyperparameter Tuning before Training...")
            self.hyperparameter_tuning(X_train, y_train)
            print(f"Optimized hyperparameters: optimizer={self.optimizer}, \
            batch_size={self.batch_size}, dropout_rate={self.dropout_rate}")

        callbacks = []
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=3, restore_best_weights=True)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.5, patience=2, min_lr=1e-5)
            callbacks = [early_stop, reduce_lr]

        print("Training the CNN model...")
        self.model.fit(X_train, y_train, epochs=min(epochs, 50), batch_size=self.batch_size,
                       validation_split=0.1, callbacks=callbacks)

    def predict(self, X_test):
        """
        Predicts labels using the trained CNN model.

        Args:
            X_test: feature matrix for testing.

        Returns:
            predicted class labels.
        """
        predictions = self.model.predict(X_test)
        return predictions.argmax(axis=1)

    def evaluate(self, X_test, y_test):
        """
        Evaluates model performance using accuracy and classification report.

        Args:
            X_test: feature matrix for testing.
            y_test: true labels for testing data.

        Returns:
            accuracy score of the model.
        """
        y_pred = self.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
        return accuracy_score(y_test, y_pred)

    def save_model(self, filename="cnn_model.h5"):
        """
        Saves the trained CNN model.

        Args:
            filename: path to save the model.

        Returns:
            None
        """
        self.model.save(filename)
        print(f"Model saved as {filename}")

    def load_model(self, filename="cnn_model.h5"):
        """
        Loads a trained CNN model.

        Args:
            filename: path to the saved model.

        Returns:
            None
        """
        try:
            self.model = keras.models.load_model(filename)
            print(f"Loaded model from {filename}")
        except FileNotFoundError:
            print(f"No saved model found at {filename}. Try training a new one.")
