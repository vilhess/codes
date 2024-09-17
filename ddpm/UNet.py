import torch 
import torch.nn as nn 


def positional_encoding(n_steps=1000, time_embedding_dim=256, n=10000):
    P = torch.zeros((n_steps, time_embedding_dim))
    for k in range(n_steps):
        for i in range(time_embedding_dim//2):
            P[k, 2*i]=torch.sin(torch.tensor(k/n**(2*i/time_embedding_dim)))
            P[k, 2*i +1]=torch.cos(torch.tensor(k/n**(2*i/time_embedding_dim)))
    return P


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    

class UNet(nn.Module):
    def __init__(self, n_steps=1000, time_embedding_dim=256):
        super(UNet, self).__init__()

        self.time_embed = nn.Embedding(n_steps, time_embedding_dim)
        self.time_embed.weight.data = positional_encoding(n_steps, time_embedding_dim)
        self.time_embed.requires_grad_ = False

        # Input image : 1, 28, 28
        self.te1 = self._make_te(time_embedding_dim, 1)
        self.b1 = nn.Sequential(
            Block(1, 10),
            Block(10, 10),
            Block(10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        # Input : 10, 14, 14
        self.te2 = self._make_te(time_embedding_dim, 10)
        self.b2 = nn.Sequential(
            Block(10, 20),
            Block(20, 20),
            Block(20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        # Input : 20, 7, 7
        self.te3 = self._make_te(time_embedding_dim, 20)
        self.b3 = nn.Sequential(
            Block(20, 40),
            Block(40, 40),
            Block(40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 4, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Input : 40, 3, 3
        self.te_mid = self._make_te(time_embedding_dim, 40)
        self.b_mid = nn.Sequential(
            Block(40, 20),
            Block(20, 20),
            Block(20, 40)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 4, 1),
        )

        # Input : (40, 7, 7) + (40, 7, 7) -> (80, 7, 7)
        self.te4 = self._make_te(time_embedding_dim, 80)
        self.b4 = nn.Sequential(
            Block(80, 40),
            Block(40, 20),
            Block(20, 20)
        )
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)

        # Input : (20, 14, 14) + (20, 14, 14) -> (40, 14, 14)
        self.te5 = self._make_te(time_embedding_dim, 40)
        self.b5 = nn.Sequential(
            Block(40, 20),
            Block(20, 10),
            Block(10, 10)
        )
        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)

        # Input : (10, 28, 28) + (10, 28, 28) -> (20, 28, 28)
        self.te6 = self._make_te(time_embedding_dim, 20)
        self.b6 = nn.Sequential(
            Block(20, 10),
            Block(10, 10),
            Block(10, 10)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(10, 10, 3, 1, 1),
            nn.Conv2d(10, 1, 3, 1, 1)
        )

    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)

        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))

        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))

        out6 = torch.cat((out1, self.up3(out5)), dim=1)
        out6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out6)
        return out


    def _make_te(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            nn.SiLU(),
            nn.Linear(in_features=output_dim, out_features=output_dim)
        )
    
if __name__=="__main__":
    x = torch.randn(10, 1, 28, 28)
    model = UNet()
    t = torch.ones(10, ).long()
    print(model(x, t).shape)