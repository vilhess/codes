import torch 
import torch.nn as nn 

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()

        self.Wxr = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Whr = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.br = nn.Parameter(torch.randn(hidden_size))

        self.Wxz = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Whz = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bz = nn.Parameter(torch.randn(hidden_size))

        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bh = nn.Parameter(torch.randn(hidden_size))

        self.act = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, xt, ht1):

        Rt = self.act(xt @ self.Wxr + ht1 @ self.Whr + self.br)
        Zt = self.act(xt @ self.Wxz + ht1 @ self.Whz + self.bz)
        HTt = self.tanh(xt @ self.Wxh + (Rt * ht1) @ self.Whh + self.bh)
        Ht = Zt * ht1 + (1 - Zt) * HTt
        return Ht


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstms = nn.ModuleList([])
        for layer in range(num_layers):
            lstm = GRUCell(input_dim, hidden_dim)
            self.lstms.append(lstm)
            input_dim=hidden_dim
        

    def forward(self, x):
        batch, window, _ = x.size()
        h = [torch.zeros(batch, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]

        for t in range(window):
            xt = x[:, t, :]
            for i, cell in enumerate(self.lstms):
                h[i] = cell(xt, h[i])
                xt = h[i]

        return h[-1] 
    
if __name__=="__main__":
    
    lstm = GRU(10, 100, 20)

    x = torch.randn(12, 5, 10) # batch, window, info

    out = lstm(x)
    print(out.shape)