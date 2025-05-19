import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# ========= Data Loading ========= #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

target_digit = 5  # Change this to your student ID's last digit

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
filtered_indices = [i for i, (_, label) in enumerate(dataset) if label == target_digit]
filtered_dataset = Subset(dataset, filtered_indices[:5000])
loader = DataLoader(filtered_dataset, batch_size=64, shuffle=True)

# ========= Visualize Real Digit Images ========= #
examples = next(iter(loader))[0][:32]
examples = examples * 0.5 + 0.5  # Unnormalize
grid = torchvision.utils.make_grid(examples, nrow=8, padding=2)
plt.figure(figsize=(8, 8))
plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
plt.title(f"Real MNIST Digit '{target_digit}' Only (Before Training)")
plt.axis("off")
plt.show()

# ========= Define Generator and Discriminator ========= #
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

# ========= Training Setup ========= #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
lr = 0.0005
num_epochs = 20

gen = Generator(z_dim).to(device)
disc = Discriminator().to(device)
criterion = nn.BCELoss()
opt_gen = optim.Adam(gen.parameters(), lr)
opt_disc = optim.Adam(disc.parameters(), lr)

# ========= Training Loop ========= #
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.size(0)

        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real)
        disc_fake = disc(fake.detach())
        loss_real = criterion(disc_real, real_labels)
        loss_fake = criterion(disc_fake, fake_labels)
        loss_disc = (loss_real + loss_fake) / 2

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        output = disc(fake)
        loss_gen = criterion(output, real_labels)  # Trick the disc to believe fake is real

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

    # Visualize generated samples
    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
        gen.eval()
        with torch.no_grad():
            sample_noise = torch.randn(32, z_dim).to(device)
            fake_images = gen(sample_noise).view(-1, 1, 28, 28)
            fake_images = fake_images * 0.5 + 0.5  # Unnormalize
            grid = torchvision.utils.make_grid(fake_images.cpu(), nrow=8, padding=2)
            plt.figure(figsize=(8, 8))
            plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
            plt.title(f"Generated Digit '{target_digit}' Samples - Epoch {epoch+1}")
            plt.axis("off")
            plt.show()
        gen.train()
