import torch 
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from model import VAE

DEVICE="mps"
model = VAE(in_dim=28*28, hidden_dims=[512, 256], latent_dim=2).to(DEVICE)

checkpoint = torch.load('checkpoint/model.pkl')
model.load_state_dict(checkpoint['state_dict'])

decoder = model.decoder

xs = torch.linspace(-10, 10, 50)
ys = torch.linspace(-10, 10, 50)

grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
Zs = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=1).to(DEVICE)

with torch.no_grad():
    reconstructed = decoder(Zs)
reconstructed = reconstructed.view(-1, 1, 28, 28)

grid_img = make_grid(reconstructed, nrow=50, normalize=True)

plt.figure(figsize=(10, 10))
plt.imshow(grid_img.cpu().permute(1, 2, 0).detach(), cmap="gray", extent=(-10, 10, -10, 10))

if not os.path.isdir('figures'):
    os.mkdir("figures")
plt.savefig("figures/generated.png")