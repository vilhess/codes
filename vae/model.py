import torch 
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, x):
        h = self.seq(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, out_dim):
        super(Decoder, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], out_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        reconstructed = self.seq(z)
        return reconstructed
    

class VAE(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_dim=in_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dims=hidden_dims[::-1], out_dim=in_dim)

    def _rep_trick(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        return mu + torch.sqrt(logvar.exp())*epsilon
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self._rep_trick(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar
    
if __name__=="__main__":
    x = torch.randn(10, 1, 28, 28)
    model = VAE(in_dim=28*28, hidden_dims=[512, 256], latent_dim=10)

    x_input = x.flatten(start_dim=1)
    print(f"Input Image shape : {x_input.shape}")

    rec, mu, logvar = model(x_input)
    print(f"Generated shape : {rec.shape}")
    print(f"Latent space shape : {mu.shape}, {logvar.shape}")