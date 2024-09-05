import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from model import VAE



DEVICE = "cpu"
model = torch.load("checkpoint/model.pkl").to(DEVICE)

decoder = model.decoder

z = torch.randn(100, 2).to(device=DEVICE)
out = decoder(z)
out = out.view(-1, 1, 28, 28)

grid_image = make_grid(out, nrow=10, normalize=True)

plt.figure(figsize=(10, 10))
plt.imshow(grid_image.cpu().permute(1, 2, 0).detach(), cmap="gray")
if not os.path.isdir('figures'):
    os.mkdir('figures')
plt.savefig('figures/generated.png')
plt.axis('off')