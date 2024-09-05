import torch 
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.input_hidden = nn.Linear(in_dim, hidden_dim)
        self.hidden_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_sigma = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.input_hidden(x)
        h = self.relu(h)

        mu = self.hidden_mu(h)
        sigma = self.hidden_sigma(h)

        return mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super(Decoder, self).__init__()

        self.latent_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_out  = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.latent_hidden(z)
        h = self.relu(h)
        out = self.sigmoid(self.hidden_out(h))
        return out
    
class VAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, in_dim)

    def rep_trick(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma*epsilon
        return z
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.rep_trick(mu, sigma)
        out = self.decoder(z)
        return out, mu, sigma
    
if __name__=="__main__":
    x = torch.randn((10, 1, 28, 28))
    model = VAE(28*28, 128, 10)

    x_input = torch.flatten(x, start_dim=1)
    print(f"Input image shape : {x_input.shape}")

    x_output, mu, sigma = model(x_input)

    print(f"Generated image shape : {x_output.shape}")
    print(f"Latent space shape : {mu.shape}, {sigma.shape}")

