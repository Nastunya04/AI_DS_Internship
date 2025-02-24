# **Animal Classification with ResNet-50**
This project implements an **image classification model** for recognizing animal species using a **pretrained ResNet-50**. 
The model is trained on a dataset containing **15 different animal classes**.

---
## **Project Structure**
```bash
📂 TASK2/
│── train.py                # Script for training the ResNet-50 model
│── inference.py            # Script for performing inference on new images
│── task2.ipynb             # Jupyter Notebook for dataset analysis and visualization
│── best_model.pth          # Saved trained model weights
│── requirements.txt        # List of required dependencies
│── test_terminal/          # Folder containing test images
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
📂 animal_data.zip          # Compressed dataset (must be unzipped before use)
```
## Installation and Setup
### Install Dependencies
Ensure that you have Python 3.8+ installed, then install the required packages:
```bash
pip install -r requirements.txt
```
### Extract the Dataset
Before training, you must unzip the dataset:
```bash
unzip animal_data.zip -d animal_data(or tar -xf animal_data.zip -C animal_data on Windows)
```
## Training the Model
To train the ResNet-50 model from scratch, run:
```bash
python train.py --data_dir animal_data --num_epochs 10 --batch_size 32 --lr 1e-4 --use_cuda
```
⚠️ Warning: Training from scratch can be time-consuming. A pre-trained model is available as best_model.pth.

## Running Inference
Instead of training from scratch, use the pre-trained model in inference.py:
```bash
python inference.py --image_path test_terminal/1.jpg --use_cuda
```
## EDA
To explore the dataset distribution and visualize images, use the Jupyter Notebook *task2.ipynb*

## About the NER part
This task primarily focused on image classification, but it also included a Named Entity Recognition (NER) component. 
Due to the lack of experience in the NLP field, 
I was unable to complete the NER part, but I look forward to exploring it further as I expand my knowledge.
