import torch 
import torch.nn as nn 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import trange
import os
from model import VAE
from loss import LossVAE

EPOCHS=20
BATCH_SIZE=128
LR=3e-4
DEVICE="mps"

dataset = MNIST(root="../Dataset/", train=True, download=False, transform=ToTensor())
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VAE(in_dim=28*28, hidden_dims=[512, 256], latent_dim=2).to(DEVICE)
criterion = LossVAE()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

pbar = trange(EPOCHS, desc="Training")
for epoch in pbar:
    epoch_loss = 0

    for images, labels in trainloader:
        inputs = images.flatten(start_dim=1).to(DEVICE)
        reconstructed, mu, logvar = model(inputs)
        loss = criterion(inputs, reconstructed, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()

    epoch_loss = epoch_loss/len(trainloader)
    pbar.set_description(f"epoch : {epoch} ; loss : {epoch_loss}")

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
checkpoint = {"state_dict":model.state_dict()}
torch.save(checkpoint, "checkpoint/model.pkl")