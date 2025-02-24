"""
main.py

This script is responsible for running the training process with tune=True \
to find the best hyperparameters.
Once trained, the models are saved so they can be loaded and reused later \
without retraining.

WARNING: This script will take a long time to execute!

If you just want to load and test models, use `demo.ipynb` instead.
"""

from tensorflow import keras
from mnist_classifier import MnistClassifier
def preprocess_data():
    """
    Loads and preprocesses the MNIST dataset for different classifiers.

    Returns:
        X_train_rf, X_test_rf: flattened images for Random Forest.
        X_train_nn, X_test_nn: normalized flattened images for FNN.
        X_train_cnn, X_test_cnn: normalized and reshaped images for CNN.
        y_train, y_test: labels.
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # data for Random Forest
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_test_rf = X_test.reshape(X_test.shape[0], -1)

    # data for FNN
    X_train_nn = X_train_rf / 255.0
    X_test_nn = X_test_rf / 255.0

    # data for CNN
    X_train_cnn = (X_train / 255.0).reshape(-1, 28, 28, 1)
    X_test_cnn = (X_test / 255.0).reshape(-1, 28, 28, 1)

    return (X_train_rf, X_train_nn, X_train_cnn, y_train), \
    (X_test_rf, X_test_nn, X_test_cnn, y_test)


def train_and_evaluate(algorithm, X_train, X_test, y_train, y_test, model_filename, **kwargs):
    """
    Trains, evaluates, saves, and loads an MNIST classifier with customizable parameters.

    Args:
        algorithm: 'rf', 'nn', or 'cnn'.
        X_train: training feature set.
        X_test: testing feature set.
        y_train: training labels.
        y_test: testing labels.
        model_filename: file path to save/load the model.
        kwargs: additional parameters for the selected classifier.
    """
    print(f"\n--- Training {algorithm.upper()} Model ---")
    classifier = MnistClassifier(algorithm, **kwargs)

    classifier.train(X_train, y_train)

    accuracy = classifier.evaluate(X_test, y_test)
    print(f"{algorithm.upper()} Accuracy: {accuracy:.4f}")

    classifier.save_model(model_filename)

    print(f"\n--- Loading {algorithm.upper()} Model and Re-evaluating ---")
    classifier.load_model(model_filename)
    loaded_accuracy = classifier.evaluate(X_test, y_test)
    print(f"Restored {algorithm.upper()} Accuracy: {loaded_accuracy:.4f}")


def main():
    """
    Main function to train and evaluate all MNIST classifiers with customizable parameters.
    """
    (X_train_rf, X_train_nn, X_train_cnn, y_train), \
    (X_test_rf, X_test_nn, X_test_cnn, y_test) = preprocess_data()


    train_and_evaluate("rf", X_train_rf, X_test_rf, y_train, y_test,
                       model_filename="random_forest_model.pkl",
                       tune=True)

    train_and_evaluate("nn", X_train_nn, X_test_nn, y_train, y_test,
                       model_filename="feedforward_nn_model.h5",
                       tune=True)

    train_and_evaluate("cnn", X_train_cnn, X_test_cnn, y_train, y_test,
                       model_filename="cnn_model.h5",
                       tune=True)

if __name__ == "__main__":
    main()
