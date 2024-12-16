import torch 
import torch.nn as nn 

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ConvGRUCell, self).__init__()

        self.conv_xr = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_hr = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_xz = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_hz = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)

        self.conv_xh = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv_hh = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same", bias=True)

        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, xt, ht1):

        Rt = self.act(self.bn(self.conv_xr(xt) + self.conv_hr(ht1)))
        Zt = self.act(self.bn(self.conv_xz(xt) + self.conv_hz(ht1)))
        HTt = self.tanh(self.bn(self.conv_xh(xt) + self.conv_hh(Rt * ht1)))
        Ht = Zt * ht1 + (1 - Zt) * HTt
        return Ht
    

class ConvGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(ConvGRU, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.lstms = nn.ModuleList([])
        for layer in range(num_layers):
            lstm = ConvGRUCell(in_channels, hidden_channels)
            self.lstms.append(lstm)
            in_channels=hidden_channels
        

    def forward(self, x):
        batch, times, _, height, width = x.size()
        h = [torch.zeros(batch, self.hidden_channels, height, width).to(x.device) for _ in range(self.num_layers)]

        for t in range(times):
            xt = x[:, t, :, :, :]
            for i, cell in enumerate(self.lstms):
                h[i] = cell(xt, h[i])
                xt = h[i]

        return h[-1] 
    
if __name__=="__main__":
    
    cgru = ConvGRU(2, 10, 20)

    x = torch.randn(20, 10, 2, 32, 32) # batch, times, channels, height, width

    out = cgru(x)
    print(out.shape)