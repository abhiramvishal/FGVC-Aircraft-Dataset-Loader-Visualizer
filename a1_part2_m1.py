# a1_part2_m1.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision.models import ResNet50_Weights

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir = r'fgvc-aircraft-2013b\data\images'
train_file = r'fgvc-aircraft-2013b\data\images_variant_train.txt'
test_file = r'fgvc-aircraft-2013b\data\images_variant_test.txt'

# Load image labels
def load_image_labels(file_path):
    image_ids, variants = [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_ids.append(parts[0])
            variants.append(" ".join(parts[1:]))
    df = pd.DataFrame({'image_id': image_ids, 'variant': variants})
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    return df

train_df = load_image_labels(train_file)
test_df = load_image_labels(test_file)

# Custom dataset
class AircraftDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.loader = default_loader
        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(self.df['variant'].unique()))}
        self.df['label'] = self.df['variant'].map(self.class_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.loader(self.df.loc[idx, 'image_path'])
        label = self.df.loc[idx, 'label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset & Dataloaders
train_dataset = AircraftDataset(train_df, transform=transform)
test_dataset = AircraftDataset(test_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
num_classes = len(train_dataset.class_to_idx)

# Get model function
def get_resnet_model(num_classes, fine_tune=False):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Training
def train_model(model, dataloader, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Testing
def test_model(model, dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Run Transfer Learning
print("=== TRANSFER LEARNING ===")
model_transfer = get_resnet_model(num_classes, fine_tune=False).to(device)
train_model(model_transfer, train_loader, num_epochs=5)
test_model(model_transfer, test_loader)

# Run Fine-Tuning
print("\n=== FINE-TUNING ===")
model_finetune = get_resnet_model(num_classes, fine_tune=True).to(device)
train_model(model_finetune, train_loader, num_epochs=5)
test_model(model_finetune, test_loader)