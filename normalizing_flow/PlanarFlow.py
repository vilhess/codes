import torch 
import torch.nn as nn 

class PlanarTransform(nn.Module):
    def __init__(self, dim=20, act=torch.tanh):
        super(PlanarTransform, self).__init__()

        self.w = nn.Parameter(torch.randn(dim)*0.01)
        self.b = nn.Parameter(torch.randn(1)*0.01)
        self.u = nn.Parameter(torch.randn(dim)*0.01)
        self.act = act

        if self.act == torch.tanh:
            self.derivative = lambda z: 1 - self.act(z)**2
        
        else:
            self.derivative = None
            print('Not implemented for this activation')

    def m(self, x):
        return -1 + torch.log(1+torch.exp(x))
    
    def u_hat(self, u, w):
        return u + (self.m(torch.matmul(w, u)) - torch.matmul(w, u)) * w/(torch.norm(w, p=2)**2)
    
    def forward(self, z, logdet=True):
        affine = torch.einsum("D, BD ->B", self.w, z) + self.b
        affine = affine.unsqueeze(1)
        u_hat = self.u_hat(self.u, self.w)
        z = z + u_hat*self.act(affine)

        if logdet:
            phi = self.w * self.derivative(affine)
            logdet = - torch.log(torch.abs(1+torch.einsum("BD, D -> B", phi, u_hat))+1e-8)
            return z, logdet.unsqueeze(1)
        else:
            return z 
        


class PlanarFlow(nn.Module):
    def __init__(self, dim=20, act=torch.tanh, num_steps=20):
        super(PlanarFlow, self).__init__()

        self.flows = nn.ModuleList([])
        for _ in range(num_steps):
            tr = PlanarTransform(dim=dim, act=act)
            self.flows.append(tr)

    def forward(self, z, logdet=True):
        sum_logdet=0
        if logdet:
            for flow in self.flows:
                z, logdet_value = flow(z, logdet)
                sum_logdet+=logdet_value
            return z, sum_logdet
        else:
            for flow in self.flows:
                z = flow(z, logdet)
            return z

if __name__=="__main__":

    z = torch.randn(10, 20)
    pf = PlanarFlow(dim=20, act=torch.tanh, num_steps=20)

    zk, sum_logdet = pf(z, logdet=True)  
    print(f"Shape of z_k : {zk.shape}") # (10, 20)
    print(f"Shape of sum logdet : {sum_logdet.shape}")  # (10, 1)