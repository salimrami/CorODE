
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

class CortexODE(nn.Module):
    """
    The deformation network of CortexODE model.
    """

    def __init__(self, dim_in=3, dim_h=128, kernel_size=5, n_scale=3):
        super(CortexODE, self).__init__()

        self.C = dim_h  # hidden dimension
        self.K = kernel_size  # kernel size
        self.Q = n_scale  # number of scales

        # FC layers
        self.fc1 = nn.Linear(dim_in, self.C)
        self.fc2 = nn.Linear(self.C * 2, self.C * 4)
        self.fc3 = nn.Linear(self.C * 4, self.C * 2)
        self.fc4 = nn.Linear(self.C * 2, dim_in)

        # local convolution
        self.localconv = nn.Conv3d(self.Q, self.C, (self.K, self.K, self.K))
        self.localfc = nn.Linear(self.C, self.C)

        # for cube sampling
        self.initialized = False
        self.x_shift = None
        self.cubes = None

    def _initialize(self, V):
        # initialize coordinates shift and cubes
        self.x_shift = self.x_shift.to(V.device)
        self.cubes = self.cubes.to(V.device)
        self.initialized = True

    def set_data(self, x, V):
        # x: coordinates
        # V: input brain MRI volume
        if V.dim() == 4:
            D1, D2, D3 = V.shape[1:]  # Extract dimensions from the tensor shape
        elif V.dim() == 3:
            D1, D2, D3 = V.shape
        else:
            raise ValueError("Invalid input tensor dimensions")

        if not self.initialized:
            self.x_shift = torch.Tensor(np.linspace(-self.K // 2, self.K // 2, self.K)).to(V.device)
            grid_3d = torch.stack(torch.meshgrid(self.x_shift, self.x_shift, self.x_shift), dim=0).permute(2, 1, 3, 0)
            self.x_shift = grid_3d.reshape(-1, 3)
            self.cubes = torch.zeros([1, self.Q, self.K, self.K, self.K]).to(V.device)
            self._initialize(V)

        # Verify the shape of V
        if len(V.shape) != 3:
            raise ValueError("Invalid shape of V. Expected a 3-dimensional volume.")

        D1, D2, D3 = V.shape
        print("V shape:", V.shape)
        print("V[0, 0] shape:", V[0, 0].shape)

        # set the shape of the volume
        D1, D2, D3 = V.shape[0], V.shape[1], V.shape[2]
        D = max([D1, D2, D3])
        # rescale for grid sampling
        self.rescale = torch.Tensor([D3 / D, D2 / D, D1 / D]).to(V.device)
        self.D = D

        self.m = x.shape[1]  # number of points
        self.neighbors = self.cubes.repeat(self.m, 1, 1, 1, 1)  # repeat m cubes

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
                xq = x.unsqueeze(-2) + self.x_shift / self.D * 2 * (2 ** q)
                xq = xq.contiguous().view(1, -1, 3).unsqueeze(-2).unsqueeze(-2)
                xq = xq / self.rescale  # rescale the coordinates
                # sample the q-th cube
                vq = F.grid_sample(self.Vq[q], xq, mode='bilinear', padding_mode='border', align_corners=True)
                # update the cubes
                self.neighbors[:, q] = vq[0, 0].view(self.m, self.K, self.K, self.K)

        return self.neighbors.clone()
