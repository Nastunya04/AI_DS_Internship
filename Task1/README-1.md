# MNIST Classification with Random Forest, Feed-Forward Neural Network, and CNN
This project implements three different classifiers to recognize handwritten digits from the MNIST dataset:
- Random Forest (RF)
- Feed-Forward Neural Network (FNN)
- Convolutional Neural Network (CNN)

The models are trained, tuned, and evaluated using different preprocessing techniques.

---
# The project structure
```bash
📂 Task1/
│── main.py                 # Main script for training and saving models with hyperparameter tuning
│── demo.ipynb              # Jupyter Notebook for interactively testing models and edge cases
│── requirements.txt        # List of required dependencies
│── mnist_classifier.py     # Unified wrapper (MnistClassifier) for handling all models
│── random_forest.py        # Implements the Random Forest classifier
│── feedforward_nn.py       # Implements the Feed-Forward Neural Network
│── cnn.py                  # Implements the Convolutional Neural Network
│── interface.py            # Abstract base class (MnistClassifierInterface) for models
```
```bash
📂 saved_models/            # Stores pre-trained models for reuse
│── random_forest_model.pkl # Pre-trained Random Forest model
│── feedforward_nn_model.h5 # Pre-trained FNN model
│── cnn_model.h5            # Pre-trained CNN model
```
## Installation and Setup

### Install Dependencies
Ensure that you have Python 3.8+ installed and run:

```bash
pip install -r requirements.txt
```
### Tune the models(optional)
If you want to tune the models, you can run the **main.py** file
```bash
python main.py
```
⚠️ Warning: Running main.py will train all models with tune=True, which can be time-consuming.
Pre-trained models are available in the saved_models/ folder
### Run the interactive demo
Instead of training from scratch, you can use the pre-trained models in demo.ipynb:\
	•	Load pre-trained models\
	•	Evaluate their performance\
	•	Test them on edge cases (e.g., noisy, occluded, rotated images)\
  • Train and test models with default parameters

## Usage
#### Loading dataset
```bash
from mnist_classifier import MnistClassifier
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)
```
#### Training a Specific Model

You can train an individual model without tuning:
```bash
from mnist_classifier import MnistClassifier
rf_classifier = MnistClassifier("rf")
rf_classifier.train(X_train_rf, y_train)
```
#### Loading Pre-Trained Models
Instead of training from scratch, load an existing model:
```bash
rf_classifier.load_model("saved_models/random_forest_model.pkl")
```
#### Evaluating a Model
```bash
accuracy = rf_classifier.evaluate(X_test_rf, y_test)
print(f"Random Forest Accuracy: {accuracy:.4f}")
```
