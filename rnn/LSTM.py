import torch 
import torch.nn as nn 

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        self.Wxi = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Whi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.randn(hidden_size))

        self.Wxf = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Whf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.randn(hidden_size))

        self.Wxo = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Who = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.randn(hidden_size))

        self.Wxc = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Whc = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bc = nn.Parameter(torch.randn(hidden_size))

        self.act = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, xt, ht1, ct1):

        It = self.act( xt @ self.Wxi + ht1 @ self.Whi + self.bi )
        Ft = self.act( xt @ self.Wxf + ht1 @ self.Whf + self.bf )
        Ot = self.act( xt @ self.Wxo + ht1 @ self.Who + self.bo )
        CTt = self.tanh( xt @ self.Wxc + ht1 @ self.Whc + self.bc )

        Ct = Ft*ct1 + It*CTt 
        Ht = Ot*self.tanh(Ct)

        return Ht, Ct
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstms = nn.ModuleList([])
        for layer in range(num_layers):
            lstm = LSTMCell(input_dim, hidden_dim)
            self.lstms.append(lstm)
            input_dim=hidden_dim
        

    def forward(self, x):
        batch, window, _ = x.size()
        h = [torch.zeros(batch, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]

        for t in range(window):
            xt = x[:, t, :]
            for i, cell in enumerate(self.lstms):
                h[i], c[i] = cell(xt, h[i], c[i])
                xt = h[i]

        return h[-1] 
    
if __name__=="__main__":
    
    lstm = LSTM(10, 100, 20)

    x = torch.randn(12, 5, 10) # batch, window, info

    out = lstm(x)
    print(out.shape)