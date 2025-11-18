#Physics informed neural network
#5 Steps: Define V as a parametric function, build the Hamiltonian as a tensor
#Solve the loss function and backpropagate, train the model on initial parameters
#Plot the band structure and demonstrate a correct gap/recovery of EPM-like results

##################################################################################

import torch 
import torch.nn as nn
import torch.optim as optim
# Defining V(G): physics informed part + NN part
# V(G) = w_0+w_1|G|^2*exp(-(|G|/sigma)**2) + NN(G)

class PINN_V(nn.Module):
    def __init__(self):
        super().__init__()
        #params for the physics based part
        self.w0 = nn.Parameter(torch.tensor(-0.2))
        self.w1 = nn.Parameter(torch.tensor(0.05))
        self.w2 = nn.Parameter(torch.tensor(-0.1))
        self.sigma = nn.Parameter(torch.tensor(1.0))
        
        #NN part
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, G):
        # G = tensor of shape (..., 3)
        #returns V, a scalar
        G_norm = torch.linalg.norm(G, dim=-1) #computing the norm
        # Physics based part
        #w0 represents a constant offset, this more or less just sets the zero energy of the potential
        #w1 affects the speed at which the potential grows with tge g vector
        #sigma sets the strength of a gaussian decay factor
        #these are all optimised during training and contribute to the loss
        V_phys = self.w0 + self.w1 * G_norm**2 + self.w2 * torch.exp(-(G_norm/self.sigma)**2)
        # NN correction
        V_corr = torch.tanh(self.net(G).squeeze(-1)) * 5.0 
        return V_phys + V_corr
