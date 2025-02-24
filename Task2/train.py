"""Parametrized train.py file for the Image Classification model"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

def train(args):
    """Trains ResNet-50 on the dataset and evaluates the model."""
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    basic_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=basic_transforms)
    dataset_size = len(full_dataset)
    print(f"Total images: {dataset_size}")

    train_size = int(args.train_ratio * dataset_size)
    val_size = int(args.val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(full_dataset, \
    [train_size, val_size, test_size])
    print(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}, \
    Test size: {len(test_subset)}")

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, \
    shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_subset,   batch_size=args.batch_size, \
    shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_subset,  batch_size=args.batch_size, \
    shuffle=False, num_workers=2)

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, args.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def train_one_epoch(model, optimizer, criterion, dataloader, device):
        """Trains the model for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate(model, criterion, dataloader, device):
        """Evaluates the model on validation or test set."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    best_val_acc = 0.0
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)

        print(f"Epoch [{epoch+1}/{args.num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_path)
            print("  -> Best model saved.")

    print("Training complete.")

    model.load_state_dict(torch.load(args.model_path))
    test_loss, test_acc = evaluate(model, criterion, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train a ResNet-50 for Image Classification')
    parser.add_argument('--data_dir', type=str, default='animal_data', help='Directory of dataset')
    parser.add_argument('--model_path', type=str, default='best_model.pth', \
    help='Path to save best model')
    parser.add_argument('--num_classes', type=int, default=15, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
