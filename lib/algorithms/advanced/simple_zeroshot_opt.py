import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from lib.algorithms.advanced.utils import quaternion_to_matrix


class RotOpt(nn.Module):
    def __init__(self, batch_size=100, axis='y', minT=0.5, maxT=2):
        super().__init__()
        self.rot_vect = nn.Parameter(torch.ones((batch_size, 1)))
        for axe in axis:
            setattr(self, 'rot_vect_%s' % axe, nn.Parameter(torch.zeros((batch_size, 1))))
        self.identity = nn.Parameter(torch.eye(3), requires_grad=False)
        self.batch_size = batch_size
        self.scale = nn.Parameter(torch.ones((batch_size, 1, 1)))
        self.minT = minT
        self.maxT = maxT
    
    def forward(self, x, T, K):
        rot_mat = self.generate_matrix()
        x = rot_mat.bmm(x.permute(0, 2, 1)) + (T * torch.clamp(self.scale, self.minT, self.maxT)).permute(0, 2, 1)
        x = K.bmm(x).permute(0, 2, 1)
        x = x[:, :, :2] / x[:, :, 2:]
        return x
       
    def generate_matrix(self):
        return quaternion_to_matrix(torch.cat([self.rot_vect, 
                                               getattr(self, 'rot_vect_x', torch.zeros((self.batch_size, 1), device=self.rot_vect.device)),
                                               getattr(self, 'rot_vect_y', torch.zeros((self.batch_size, 1), device=self.rot_vect.device)),
                                               getattr(self, 'rot_vect_z', torch.zeros((self.batch_size, 1), device=self.rot_vect.device))], dim=-1))
    
def perpendicular_distance(point, vector):
    # Compute the projection of the vector_to_point onto the vector
    projection = torch.sum(point * vector, dim=-1, keepdim=True) * vector
    return projection - point

def error_compute(key2d,key3d,K):
    Kinv = torch.inverse(K)
    key2d = torch.cat([key2d, torch.ones(key2d.shape[0], key2d.shape[1], 1)], dim=-1).permute(0, 2, 1)
    ray2d = Kinv.bmm(key2d).permute(0, 2, 1)
    b = ray2d - key3d
    
    return torch.max(torch.norm(b, dim=-1))

def gradient_field_gen(key2d, key3d, K, noise_type=None, t=None, conf=None, returnT=False,norm_true=None, previous_T = None):
    std = 0.0001
    '''
    Generate the gradient of 3D keypoints based on the 2D keypoints
    input and camera intrinsic.
    
    key2d: 2D keypoints torch.FloatTensor b * n * 2
    key3d: 3D keypoints torch.FloatTensor b * n * 3
    conf: 2D keypoints confidence torch.FloatTensor b * n
    K: Intrinsic parameters torch.FloatTensor b * 3 * 3
    '''
    
    # Minimize reprojection error
    device = key2d.device
    
    Kinv = torch.inverse(K)
    # key2d = key2d.permute(0, 2, 1)
    key2d = torch.cat([key2d, torch.ones(key2d.shape[0], key2d.shape[1], 1).to(device)], dim=-1)
    if conf is not None:
        conf[conf > 1] = 1
        conf[conf < 1e-4] = 1e-4
        # key2d[:, :, 2] = conf
        # key2d[:, :, :2] += torch.randn_like(key2d[:, :, :2]) * (1 - conf[:, :, None]) * 2
    key2d = key2d.permute(0, 2, 1)
    ray2d = Kinv.bmm(key2d).permute(0, 2, 1)
    ray2d = ray2d / ray2d[:, :, 2:]
    # import ipdb; ipdb.set_trace()
    if t is None:
        A = torch.zeros((key3d.shape[0], key3d.shape[1] * 2, 3)).to(device)
        b = torch.zeros((key3d.shape[0], key3d.shape[1] * 2, 1)).to(device)
        b[:, 0::2, :] = key3d[:, :, 0:1] - key3d[:, :, 2:3] * ray2d[:, :, 0:1]
        b[:, 1::2, :] = key3d[:, :, 1:2] - key3d[:, :, 2:3] * ray2d[:, :, 1:2]
        A[:, 0::2, 0] = -1
        A[:, 0::2, 2] = ray2d[:, :, 0]
        A[:, 1::2, 1] = -1
        A[:, 1::2, 2] = ray2d[:, :, 1]
        # import ipdb; ipdb.set_trace()
        key2d = key2d.permute(0, 2, 1)
        if conf is not None:
            A[:, 0::2, :] *= conf[:, :, None] * conf[:, :, None]
            A[:, 1::2, :] *= conf[:, :, None] * conf[:, :, None]
            b[:, 0::2, :] *= conf[:, :, None] * conf[:, :, None]
            b[:, 1::2, :] *= conf[:, :, None] * conf[:, :, None]
        
        ATA = A.permute(0, 2, 1).bmm(A)
        ATb = A.permute(0, 2, 1).bmm(b)
        T = torch.inverse(ATA).bmm(ATb).permute(0, 2, 1)
        T[T[:, :, 2] < 0] = T[T[:, :, 2] < 0] * -1
        
    else:
        T = t
    
    # Compute Gradient
    ray2d = ray2d / torch.norm(ray2d, dim=-1, keepdim=True)
    
    # if noise_type ==  'raygaussian':
        
    #     noise = torch.randn(*ray2d.shape).to(ray2d.device)

    #     ray2d+=std*noise
    #     ray2d = ray2d / torch.norm(ray2d, dim=-1, keepdim=True)
    
        
    gradient = perpendicular_distance(key3d + T, ray2d)
    # gradient *= conf[:, :, None] * conf[:, :, None]
    
    if noise_type == 'gaussian':
        noise =  torch.randn(*gradient.shape).to(gradient.device)
        gradient= gradient + std*noise*t
    elif noise_type == 'uniform':
        # a range of [-0.5std, 0.5std]
        noise = (torch.randn(*gradient.shape) - 0.5).to(gradient.device)
        gradient = gradient + std*noise
    
    
    
    if returnT:
        return gradient, T
    else:
        return gradient

if __name__ == '__main__':
    key2d = np.array([
        [[100, 100], [120, 120], [140, 140], [90, 100]]
    ])
    key3d = np.array([
        [[1, 1, 3], [1.2, 1.2, 3], [1.4, 1.4, 3], [0.9, 100, 3]]
    ])
    K = np.array([
        [[1000, 0, 100],
         [0, 1000, 100],
         [0, 0, 1]]
    ])
    key3d = torch.FloatTensor(key3d)

    for i in range(10):
        grad = gradient_field_gen(torch.FloatTensor(key2d),
                                  key3d,
                                  torch.FloatTensor(K))
        
        print(f'{i}th grad is {torch.mean(torch.norm(grad, dim=-1))}')
        key3d += grad
    