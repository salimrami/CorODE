import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





class CortexODE(nn.Module):
    
    #The deformation network of CortexODE model.

    #dim_in: input dimension
    #dim_h (C): hidden dimension
    #kernel_size (K): size of convolutional kernels
    #n_scale (Q): number of scales of the multi-scale input
    
    
    def __init__(self, dim_in=3,
                       dim_h=128,
                       kernel_size=5,
                       n_scale=3):
        
        super(CortexODE, self).__init__()


        C = dim_h        # hidden dimension
        K = kernel_size  # kernel size
        Q = n_scale      # number of scales
        
        self.C = C
        self.K = K
        self.Q = Q

        # FC layers
        self.fc1 = nn.Linear(dim_in, C)
        self.fc2 = nn.Linear(C*2, C*4)
        self.fc3 = nn.Linear(C*4, C*2)
        self.fc4 = nn.Linear(C*2, dim_in)
        
        # local convolution
        self.localconv = nn.Conv3d(Q, C, (K, K, K))
        self.localfc = nn.Linear(C, C)
        
        # for cube sampling
        self.initialized = False
        grid = np.linspace(-K//2, K//2, K)
        grid_3d = np.stack(np.meshgrid(grid, grid, grid), axis=0).transpose(2,1,3,0)
        self.x_shift = torch.Tensor(grid_3d.copy()).view(-1,3)
        self.cubes = torch.zeros([1, self.Q, self.K, self.K, self.K])

    def _initialize(self, V):
        # initialize coordinates shift and cubes
        self.x_shift = self.x_shift.to(V.device)
        self.cubes = self.cubes.to(V.device)
        self.initialized == True
        
    def set_data(self, x, V):
    # x: coordinates
    # V: input brain MRI volume
        if not self.initialized:
               self._initialize(V)
        
        # set the shape of the volume
        if len(V[0, 0].shape) == 1:
            
            D = max(V[0, 0].shape)
            D1, D2, D3 = D, D, D
            V = V.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        else:
            D1, D2, D3 = V[0, 0].shape
    
    # Print the shape of V for debugging
        print("V shape:", V.shape)
        print("V[0, 0] shape:", V[0, 0].shape)
        print("D1, D2, D3:", D1, D2, D3)





        D = max([D1, D2, D3])
    # rescale for grid sampling
        self.rescale = torch.Tensor([D3/D, D2/D, D1/D]).to(V.device)
        self.D = D

        self.m = x.shape[1]    # number of points
        self.neighbors = self.cubes.repeat(self.m, 1, 1, 1, 1)    # repeat m cubes
    
        # set multi-scale volume
        self.Vq = [V]
        for q in range(1, self.Q):
        # iteratively downsampling
            self.Vq.append(F.avg_pool3d(self.Vq[-1], 2))


    def forward(self, t, x):
        
        # local feature
        z_local = self.cube_sampling(x)
        z_local = self.localconv(z_local)
        z_local = z_local.view(-1, self.m, self.C)
        z_local = self.localfc(z_local)
        
        # point feature
        z_point = F.leaky_relu(self.fc1(x), 0.2)
        
        # feature fusion
        z = torch.cat([z_point, z_local], 2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        dx = self.fc4(z)
        
        return dx
    
    def cube_sampling(self, x):
        # x: coordinates
        with torch.no_grad():
            for q in range(self.Q):
                # make sure the cubes have the same size
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2**q)
                xq = xq.contiguous().view(1,-1,3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # rescale the coordinates
                # sample the q-th cube
                vq = F.grid_sample(self.Vq[q], xq, mode='bilinear', padding_mode='border', align_corners=True)
                # update the cubes
                self.neighbors[:,q] = vq[0,0].view(self.m, self.K, self.K, self.K)
        
        return self.neighbors.clone()


