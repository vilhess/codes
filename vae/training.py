import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm 
import os
from model import VAE
from loss import LossVAE

EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE=3e-4
DEVICE = "mps"

dataset = MNIST(root="../../../coding/Dataset/", train=True, transform=ToTensor(), download=True)
trainloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VAE(in_dim=28*28, hidden_dim=[512, 256], latent_dim=2).to(DEVICE)
criterion = LossVAE()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    epoch_loss = 0

    for data in tqdm(trainloader):
        x_inputs = data[0].to(DEVICE)
        x_inputs = x_inputs.flatten(start_dim=1)
        x_output, mu, sigma = model(x_inputs)
        loss = criterion(x_inputs, x_output, mu, sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()

    epoch_loss = epoch_loss/len(trainloader)
    print(f"For epoch {epoch}, current loss is {epoch_loss}")

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(model, "checkpoint/model.pkl")