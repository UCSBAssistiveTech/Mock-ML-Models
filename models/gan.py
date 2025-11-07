import torch # type: ignore
from torch import nn # type: ignore
import pandas as pd
import numpy as np

import math
import matplotlib.pyplot as plt # type: ignore

"""
Overall, this model occasionally works well, however there is one slight
current drawbacks. This dataset is just too small. GAN's usually work
better with slightly larger datasets, as this gives them better insight
into the true hyper-dimensional shape of the data. 
"""

data = pd.DataFrame({
    "pupil_dilation": [0.30, 0.45, 0.25, 0.50, 0.33, 0.48, 0.29, 0.60, 0.31, 0.52],
    "reaction_time": [1.0, 1.6, 0.8, 1.9, 1.2, 1.7, 0.9, 2.0, 1.1, 1.8],
    "fixation_stability": [0.85, 0.40, 0.90, 0.35, 0.80, 0.45, 0.95, 0.30, 0.82, 0.38],
    "label": ["low", "high", "low", "high", "low", "high", "low", "high", "low", "high"]
})

features = torch.tensor(data[["pupil_dilation", "reaction_time", "fixation_stability"]].values, dtype=torch.float32)

# Extract original features and labels
original_features = data[["pupil_dilation", "reaction_time", "fixation_stability"]].values
original_labels = data["label"].values

input_dim = 3  # 3 features now instead of 2
latent_dim = 5  # latent noise dim for generator
batch_size = 32
epochs = 2000

train_set = [(features[i], 0) for i in range(len(features))]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Define Generator with input and output adjusted to 3 features
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 28),
            nn.ReLU(),
            nn.Linear(28, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),  # output 3 features
        )
    def forward(self, noise):
        return self.model(noise)

# Define Discriminator similarly with input_dim = 3
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 28),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(28, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x)
    
# Initialize models and optimizers
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop (simplified)
for epoch in range(epochs):
    for real_data, _ in train_loader:
        batch_size = real_data.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise).detach()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        d_optimizer.zero_grad()
        d_real = discriminator(real_data)
        d_fake = discriminator(fake_data)
        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)
        d_fake = discriminator(fake_data)
        g_loss = criterion(d_fake, real_labels)
        g_loss.backward()
        g_optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

noise = torch.randn(25, latent_dim)
generated_samples = generator(noise).detach().numpy()

gen_labels = np.random.choice(["low", "high"], size=generated_samples.shape[0])

# Define colors for plotting
colors = {
    "original_low": "blue",
    "original_high": "cyan",
    "generated_low": "red",
    "generated_high": "magenta",
}

# Plot original data points by label
plt.scatter(
    original_features[original_labels == "low", 0],
    original_features[original_labels == "low", 1],
    c=colors["original_low"],
    label="Original Low",
    alpha=0.6,
    edgecolor="k",
)
plt.scatter(
    original_features[original_labels == "high", 0],
    original_features[original_labels == "high", 1],
    c=colors["original_high"],
    label="Original High",
    alpha=0.6,
    edgecolor="k",
)

# Plot generated points by label, also on first two features
plt.scatter(
    generated_samples[gen_labels == "low", 0],
    generated_samples[gen_labels == "low", 1],
    c=colors["generated_low"],
    label="Generated Low",
    marker="x"
)
plt.scatter(
    generated_samples[gen_labels == "high", 0],
    generated_samples[gen_labels == "high", 1],
    c=colors["generated_high"],
    label="Generated High",
    marker="x"
)

plt.xlabel("Pupil Dilation")
plt.ylabel("Reaction Time")
plt.title("Original and Generated Data Points by Label")
plt.legend()
plt.show()