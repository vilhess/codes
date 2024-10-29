import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

DEVICE="mps"

model = torch.load("checkpoint/model.pkl").to(DEVICE)
encoder = model.encoder

dataset = MNIST(root="../../../coding/Dataset/", train=True, transform=ToTensor())
dic_img = {i:[] for i in range(10)}

for img, lab in dataset:
    if all([len(dic_img[i])>=100 for i in range(10)]):
        break
    if len(dic_img[lab])<100:
        dic_img[lab].append(img.view(-1))

dic_img = {i:torch.stack(dic_img[i]).to(DEVICE) for i in range(10)}

labels = {
    0:'red',
    1:'blue',
    2: 'green',
    3: 'yellow',
    4: 'purple',
    5: 'orange',
    6: 'pink',
    7: 'brown',
    8: 'gray',
    9: 'cyan'
}

with torch.no_grad():
    for lab in range(10):
        mu, sigma = encoder(dic_img[lab])
        z = model.rep_trick(mu, sigma).detach().cpu()
        plt.scatter(z[:, 0], z[:, 1], c=labels[lab], label=f"Label {lab}")
    plt.legend(title='Classes', loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title('2D Latent Space Representation')
    plt.savefig("figures/latent_space.png")