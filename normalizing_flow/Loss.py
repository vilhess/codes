import torch 
import torch.nn as nn 
from PF_VAE import PlanarFlowVAE

def compute_loss(x, x_out, sum_logdet, mu, sigma):
    rec_loss = nn.MSELoss(reduction="sum")
    rec_loss = rec_loss(x, x_out)
    kl_div = - 0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2), axis=1)
    return rec_loss/len(x) + sum_logdet.mean() + kl_div.mean()

if __name__=="__main__":
    x = torch.randn(10, 784)
    pf_vae = PlanarFlowVAE(in_dim=784, hiddens_dim=[512, 256], z_dim=20, act=torch.tanh, num_steps=20)
    x_out, sum_logdet, mu, sigma = pf_vae(x)
    loss = compute_loss(x, x_out, sum_logdet, mu, sigma)
    print(loss)