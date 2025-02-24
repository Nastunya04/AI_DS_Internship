"""Parametrized inference.py file for the Image Classification model"""
import argparse
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

def load_model(model_path, num_classes, device):
    """Loads a trained ResNet-50 model from a checkpoint file."""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device, class_names):
    """Runs inference on a single image and returns the predicted class."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # add batch dimension
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

def main():
    parser = argparse.ArgumentParser(description='Inference for Image Classification')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, default='Task2/best_model.pth', \
    help='Path to the trained model')
    parser.add_argument('--num_classes', type=int, default=15, help='Number of classes')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
                   'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

    model = load_model(args.model_path, args.num_classes, device)
    prediction = predict_image(model, args.image_path, device, class_names)
    print(f"Predicted class: {prediction}")

if __name__ == '__main__':
    main()
