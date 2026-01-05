import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

# Enable blocking for CUDA error tracking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
image_dir = r'fgvc-aircraft-2013b\data\images'
train_file = r'fgvc-aircraft-2013b\data\images_variant_train.txt'
test_file = r'fgvc-aircraft-2013b\data\images_variant_test.txt'

# Load FGVC-Aircraft labels
def load_image_labels(file_path):
    image_ids = []
    variants = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_ids.append(parts[0])
            variants.append(" ".join(parts[1:]))  # Handles multi-word class names
    df = pd.DataFrame({'image_id': image_ids, 'variant': variants})
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    return df

train_df = load_image_labels(train_file)
test_df = load_image_labels(test_file)

# Data augmentation transforms
transform_train = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.RandomCrop((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Custom dataset class
class AircraftDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.loader = lambda path: Image.open(path).convert("RGB")
        unique_variants = sorted(self.df['variant'].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(unique_variants)}
        self.df['label'] = self.df['variant'].map(self.class_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.loader(self.df.loc[idx, 'image_path'])
        label = self.df.loc[idx, 'label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Model components
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + x)

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 64),
            nn.MaxPool2d(2)
        )
        self.layer1 = nn.Sequential(
            ConvBNReLU(64, 128),
            ResidualBlock(128),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            ConvBNReLU(128, 256),
            ResidualBlock(256),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            ConvBNReLU(256, 512),
            ResidualBlock(512),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x

# Training loop
def train_model(model, dataloader, num_epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
        print(f"Epoch {epoch+1}/{num_epochs} completed.")

# Testing loop
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Final Setup and Run
train_dataset = AircraftDataset(train_df, transform=transform_train)
test_dataset = AircraftDataset(test_df, transform=transform_test)
test_dataset.class_to_idx = train_dataset.class_to_idx
test_dataset.df['label'] = test_dataset.df['variant'].map(test_dataset.class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = max(train_dataset.df['label']) + 1

model = CustomCNN(num_classes=num_classes).to(device)
train_model(model, train_loader, num_epochs=10)
test_model(model, test_loader)