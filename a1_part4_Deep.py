# Your answer here
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained CNN
model = models.vgg16(pretrained=True).features[:28].to(device).eval()  # Use layers till relu4_3

# DeepDream Core
def deep_dream(image, model, iterations=20, lr=0.01):
    image = image.clone().detach().to(device).requires_grad_(True)

    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        image.data = image.data + lr * image.grad.data
        image.grad.data.zero_()

    return image.detach()

# Image Preprocessing
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# Load a few aircraft images from your dataset folder
from glob import glob
all_images = glob(os.path.join("fgvc-aircraft-2013b", "data", "images", "*.jpg"))
sample_paths = all_images[:3]  # Use 3 for DeepDream

# Dreamify and Save
os.makedirs("deepdream_outputs", exist_ok=True)
for i, path in enumerate(sample_paths):
    img = Image.open(path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    dreamed = deep_dream(input_tensor, model, iterations=30, lr=0.02)
    dreamed_img = dreamed.squeeze().cpu().detach()
    dreamed_img = dreamed_img * 0.5 + 0.5  # unnormalize

    T.ToPILImage()(dreamed_img).save(f"deepdream_outputs/deepdream_{i+1}.png")

print(" DeepDream images saved to 'deepdream_outputs/' folder")

