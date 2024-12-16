import torch 
import torch.nn as nn 

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ConvLSTMCell, self).__init__()

        self.conv_xi = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_hi = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_xf = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_hf = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_xo = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_ho = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_xc = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_hc = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)

        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, xt, ht1, ct1):

        It = self.act(self.bn(self.conv_xi(xt) + self.conv_hi(ht1)))
        Ft = self.act(self.bn(self.conv_xf(xt) + self.conv_hf(ht1)))
        Ot = self.act(self.bn(self.conv_xo(xt) + self.conv_ho(ht1)))
        CTt = self.tanh(self.bn(self.conv_xc(xt) + self.conv_hc(ht1)))

        Ct = Ft*ct1 + It*CTt 
        Ht = Ot*self.tanh(Ct)

        return Ht, Ct
    

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(ConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.lstms = nn.ModuleList([])
        for layer in range(num_layers):
            lstm = ConvLSTMCell(in_channels, hidden_channels)
            self.lstms.append(lstm)
            in_channels=hidden_channels
        

    def forward(self, x):
        batch, times, _, height, width = x.size()
        h = [torch.zeros(batch, self.hidden_channels, height, width).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.hidden_channels, height, width).to(x.device) for _ in range(self.num_layers)]

        for t in range(times):
            xt = x[:, t, :, :, :]
            for i, cell in enumerate(self.lstms):
                h[i], c[i] = cell(xt, h[i], c[i])
                xt = h[i]

        return h[-1] 
    

if __name__=="__main__":
    
    clstm = ConvLSTM(2, 10, 20)

    x = torch.randn(20, 10, 2, 32, 32) # batch, times, channels, height, width

    out = clstm(x)
    print(out.shape)