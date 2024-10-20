import torch 
import torch.nn as nn 
from PlanarFlow import PlanarFlow

class Encoder(nn.Module):
    def __init__(self, in_dim=784, hiddens_dim=[512, 256], z_dim=20):
        super(Encoder, self).__init__()
        dims = [in_dim] + hiddens_dim
        self.layers = nn.ModuleList([])
        for i in range(len(dims)-1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.ReLU()
                )
            )
        self.fc_mu = nn.Linear(dims[-1], z_dim)
        self.fc_sigma = nn.Linear(dims[-1], z_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, z_dim=20, hiddens_dim=[256, 512], out_dim=784):
        super(Decoder, self).__init__()
        dims = [z_dim] + hiddens_dim + [out_dim]
        self.layers = nn.ModuleList([])
        for i in range(len(dims)-1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.ReLU()
                )
            )
    
    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z
    
class PlanarFlowVAE(nn.Module):
    def __init__(self, in_dim=20, hiddens_dim=[512, 256], z_dim=20, act=torch.tanh, num_steps=20):
        super(PlanarFlowVAE, self).__init__()

        self.encoder = Encoder(in_dim=in_dim, hiddens_dim=hiddens_dim, z_dim=z_dim)
        self.pf = PlanarFlow(dim=z_dim, act=act, num_steps=num_steps)
        self.decoder = Decoder(z_dim=z_dim, hiddens_dim=hiddens_dim[::-1], out_dim=in_dim)

    def rep_trick(self, mu, sigma):
        epsilon = torch.rand_like(mu)
        return mu + sigma*epsilon
    
    def forward(self, x, logdet=True):
        mu, sigma = self.encoder(x)
        z0 = self.rep_trick(mu, sigma)
        if logdet:
            zk, sum_logdet = self.pf(z0, logdet)
        else:
            zk = self.pf(z0, logdet)
        x_out = self.decoder(zk)
        if logdet:
            return x_out, sum_logdet, mu, sigma
        else:
            return x_out, mu, sigma
        
if __name__=="__main__":
    pf_vae = PlanarFlowVAE(in_dim=784, hiddens_dim=[512, 256], z_dim=20, num_steps=20, act=torch.tanh)
    x = torch.randn(10, 784)
    x_out, sum_logdet, mu, sigma = pf_vae(x)
    print(f"Reconstructed image shape : {x_out.shape}")
    print(f"Sum logdet shape : {sum_logdet.shape}")
    print(f"Z_0 distribution parameters : mu : {mu.shape}, sigma : {sigma.shape}")
        