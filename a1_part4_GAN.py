import os
import shutil
import gc
import matplotlib.pyplot as plt
import torchvision.utils as vutils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Paths
image_dir = r'fgvc-aircraft-2013b\data\images'
output_dir = r'fgvc-aircraft-2013b\data\gan_class_subset'

label_files = [
    r'fgvc-aircraft-2013b\data\images_variant_train.txt',
    r'fgvc-aircraft-2013b\data\images_variant_test.txt',
    r'fgvc-aircraft-2013b\data\images_variant_val.txt'
]

target_variant = "707-320".lower()
os.makedirs(output_dir, exist_ok=True)
count = 0

# Process each split file
for file in label_files:
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_id = parts[0]
            variant = " ".join(parts[1:]).strip().lower()
            if variant == target_variant:
                src = os.path.join(image_dir, image_id + ".jpg")
                dst = os.path.join(output_dir, image_id + ".jpg")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    count += 1

print(f"Combined Extraction Done. Variant: {target_variant}")
print(f"Total images saved: {count}")

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ==== Config ====
image_size = 64
batch_size = 16
epochs = 100
nz = 100  # Latent vector
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Paths ====
data_root = r'fgvc-aircraft-2013b\data\gan_class_subset'

# ==== Transforms ====
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ==== Dataset ====
dataset = datasets.ImageFolder(
    root=os.path.dirname(data_root),
    transform=transform
)
dataset.samples = [(os.path.join(data_root, f), 0) for f in os.listdir(data_root) if f.endswith(".jpg")]
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==== Generator ====
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x): return self.model(x)

# ==== Discriminator ====
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),   # 64 → 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # 32 → 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False), # 16 → 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),   # 8 → 5
            nn.AdaptiveAvgPool2d(1),                 # 5x5 → 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(-1)

# ==== Initialize ====
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
fixed_noise = torch.randn(10, nz, 1, 1, device=device)

optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# ==== Training Loop ====
print("Training GAN...")
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        label_real = torch.full((b_size,), 1., dtype=torch.float, device=device)
        label_fake = torch.full((b_size,), 0., dtype=torch.float, device=device)

        # Train Discriminator
        netD.zero_grad()
        output_real = netD(real_images)
        lossD_real = criterion(output_real, label_real)

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        output = netD(fake_images)
        lossG = criterion(output, label_real)
        lossG.backward()
        optimizerG.step()

    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1}/{epochs} | Loss D: {lossD.item():.4f} | Loss G: {lossG.item():.4f}")

# Generate 10 samples safely
print("\n Generating 10 synthetic images...")
netG.eval()
with torch.no_grad():
    fake_images = netG(fixed_noise[:10]).cpu()

# Clear GPU memory before plotting
torch.cuda.empty_cache()
gc.collect()

vutils.save_image(fake_images, "gan_samples.png", nrow=5, normalize=True)
print(" Images saved to gan_samples.png")