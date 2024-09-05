import torch
import torch.nn as nn

class LossVAE(nn.Module):
    def __init__(self):
        super(LossVAE, self).__init__()

        self.rec_loss = nn.BCELoss(reduction="sum")

    def kl_div(self, mu, sigma):
        loss = -torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        return loss
    
    def forward(self, x, x_output, mu, sigma):
        rec_loss = self.rec_loss(x_output, x)
        kl = self.kl_div(mu, sigma)
        return rec_loss+kl