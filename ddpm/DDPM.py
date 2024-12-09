import torch 
import torch.nn as nn
from torchvision.transforms import ToPILImage 
from torchvision.utils import make_grid
from tqdm import tqdm 

class DDPM(nn.Module):
    def __init__(self, network, n_steps=1000, min_beta=1e-4, max_beta=0.02, device="cpu"):
        super(DDPM, self).__init__()

        self.device = device 
        self.network = network.to(device)
        self.n_steps = n_steps 
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def forward(self, x0, t, epsilon=None):
        if epsilon is None:
            epsilon=torch.randn(size=x0.shape).to(self.device)
        n, _, _, _ = x0.shape

        return self.alphas_bar[t].sqrt().reshape(n, 1, 1, 1)*x0 + (1-self.alphas_bar[t]).sqrt().reshape(n, 1, 1, 1)*epsilon
    
    def backward(self, noise, t):
        return self.network(noise, t)
    
    def generate_gif(self, path):
        with torch.no_grad():
            x=torch.randn(20, 1, 28, 28).to(self.device)
            frames=[]
            for t in tqdm(list(range(self.n_steps))[::-1]):
                time_tensor = (t*torch.ones((20, 1))).to(self.device).long()
                noise = self.backward(x, time_tensor)
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alphas_bar[t]

                x = (1/alpha_t.sqrt()) * (x - (1 - alpha_t)/(1-alpha_bar_t).sqrt() * noise)

                if t>0:
                    z = torch.randn(20, 1, 28, 28).to(self.device)
                    beta_t = self.betas[t]
                    sigma_t = beta_t.sqrt()
                    x = x + sigma_t*z
                if t%10==0:
                    images = [(data - torch.min(data))/(torch.max(data)-torch.min(data)) for data in x.cpu()]
                    grid = make_grid(images, nrow=5, padding=0)
                    grid = ToPILImage()(grid)
                    frames.append(grid)
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=2,
            loop=0
        )