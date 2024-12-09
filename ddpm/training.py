import torch 
import torch.nn as nn 
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from UNet import UNet
from DDPM import DDPM

DEVICE="mps"

transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x-0.5)*2)
])

n_steps, min_beta, max_beta = 1000, 1e-4, 0.02
ddpm = DDPM(UNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE)

trainset = FashionMNIST(root="../../../coding/Dataset/", transform=transform)
dataloader = DataLoader(trainset, batch_size=64, shuffle=True)

EPOCHS=100
LEARNING_RATE=3e-4

criterion = nn.MSELoss()
optimizer = optim.Adam(ddpm.parameters(), lr=LEARNING_RATE)

best_loss = float("inf")
best_model = None
best_epoch = None

for epoch in range(EPOCHS):
    epoch_loss = 0
    for step, data in enumerate(tqdm(dataloader, leave=False, desc=f"Epoch : {epoch+1}/{EPOCHS}")):
        x = data[0].to(DEVICE)
        n = len(x)

        epsilon = torch.randn(x.shape).to(DEVICE)
        t = torch.randint(0, n_steps, (n, )).to(DEVICE)

        noisy_images = ddpm(x, t, epsilon)
        epsilon_theta = ddpm.backward(noisy_images, t.reshape(n, -1))

        loss = criterion(epsilon_theta, epsilon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()*len(x)/len(dataloader.dataset)
    
    log_string = f"Loss at epoch {epoch+1}: {epoch_loss:.3f}"


    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch=epoch
        best_model=deepcopy(ddpm.state_dict())
        checkpoints = {
            "model": ddpm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch":epoch,
            "best_loss":best_loss
        }
        torch.save(checkpoints, "checkpoints/best_model.pkl")
        log_string+="--> model saved"
        ddpm.generate_gif(f"gifs/gif_{epoch}.gif")
    print(log_string)