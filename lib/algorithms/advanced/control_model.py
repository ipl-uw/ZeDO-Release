# from curses import reset_shell_mode
import os
from re import I
# from this import d
import time
import math
import copy
import pickle
import argparse
import functools
from collections import deque

import numpy as np
from scipy import integrate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
from torch import autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class ClassifierFreeSampler(nn.Module):
    def __init__(self, model, w):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.w = w  # guidance stength, 0: no guidance, [0, 4.0] in original paper

    def forward(self, batch, t, condition):
        """
        batch: [B, j, 3] or [B, j, 1]
        t: [B, 1]
        condition: [B, j, 2]
        Return: [B, j, 3] or [B, j, 1] same dim as batch
        """
        out = self.model(batch, t, condition)
        # TODO: fine-grained zero-out
        zeros = torch.zeros_like(condition)
        out_uncond = self.model(batch, t, zeros)
        return out + self.w * (out - out_uncond)


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

    return sigmas


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Control_ScoreModelFC_Adv(nn.Module):
    """
    Independent condition feature projection layers for each block
    """
    def __init__(self, config,
        n_joints=17, joint_dim=3, hidden_dim=64, embed_dim=32, cond_dim=2,
        n_blocks=2,model=None):
        super(Control_ScoreModelFC_Adv, self).__init__()

        self.config = config
        self.n_joints = n_joints
        self.joint_dim = joint_dim
        self.n_blocks = n_blocks
        
        self.act = nn.SiLU()

        self.pre_dense= nn.Linear(n_joints * joint_dim, hidden_dim)
        self.pre_dense_t = nn.Linear(embed_dim, hidden_dim)
        # self.pre_dense_cond = nn.Linear(hidden_dim, hidden_dim)
        self.pre_gnorm = nn.GroupNorm(32, num_channels=hidden_dim)
        self.dropout = nn.Dropout(p=0.25)
        self.infant_cond = nn.Parameter(torch.randn(n_joints * joint_dim),requires_grad = True)

        # time embedding
        # self.zc_layer_1 = nn.Linear(n_joints*joint_dim,n_joints*joint_dim)
        self.zc_layer_1 = nn.Linear(n_joints * joint_dim,n_joints*joint_dim)
        
        self.zc_layer_2 = nn.Linear(hidden_dim,hidden_dim)
        
        for idx in range(n_blocks):
            setattr(self, f'zc_b{idx+1}_1', nn.Linear(hidden_dim, hidden_dim))
            
            setattr(self, f'zc_b{idx+1}_2', nn.Linear(hidden_dim, hidden_dim))
        self.time_embedding_type = config.model.embedding_type.lower()
        if self.time_embedding_type == 'fourier':
            self.gauss_proj = GaussianFourierProjection(embed_dim=embed_dim)
        elif self.time_embedding_type == 'positional':
            self.posit_proj = functools.partial(get_timestep_embedding, embedding_dim=embed_dim)
        else:
            assert 0

        self.shared_time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
        )
        self.register_buffer('sigmas', torch.tensor(get_sigmas(config)))

        # conditional embedding
        # self.cond_embed = nn.Sequential(
        #     nn.Linear(n_joints * cond_dim, hidden_dim),
        #     self.act
        # )
        setattr(self, f'pre_dense_copy', nn.Linear(n_joints * joint_dim, hidden_dim))
        setattr(self, f'pre_dense_t_copy', nn.Linear(embed_dim, hidden_dim))
        # setattr(self, f'b{idx+1}_dense1_cond', nn.Linear(hidden_dim, hidden_dim))
        setattr(self, f'pre_gnorm_copy', nn.GroupNorm(32, num_channels=hidden_dim))

        for idx in range(n_blocks):
            setattr(self, f'b{idx+1}_dense1', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense1_t', nn.Linear(embed_dim, hidden_dim))
            # setattr(self, f'b{idx+1}_dense1_cond', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_gnorm1', nn.GroupNorm(32, num_channels=hidden_dim))

            setattr(self, f'b{idx+1}_dense2', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense2_t', nn.Linear(embed_dim, hidden_dim))
            # setattr(self, f'b{idx+1}_dense2_cond', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_gnorm2', nn.GroupNorm(32, num_channels=hidden_dim))
            
            setattr(self, f'b{idx+1}_dense1_copy', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense1_t_copy', nn.Linear(embed_dim, hidden_dim))
            # setattr(self, f'b{idx+1}_dense1_cond', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_gnorm1_copy', nn.GroupNorm(32, num_channels=hidden_dim))

            setattr(self, f'b{idx+1}_dense2_copy', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense2_t_copy', nn.Linear(embed_dim, hidden_dim))
            # setattr(self, f'b{idx+1}_dense2_cond', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_gnorm2_copy', nn.GroupNorm(32, num_channels=hidden_dim))

        self.post_dense = nn.Linear(hidden_dim, n_joints * joint_dim)

        # self.marginal_prob_std = marginal_prob_std_func
        self.cond_pose_mask_prob = config.training.cond_pose_mask_prob
        self.cond_part_mask_prob = config.training.cond_part_mask_prob
        self.cond_joint_mask_prob = config.training.cond_joint_mask_prob

        if self.cond_part_mask_prob > 0:
            self.part_mask = self.generate_part_mask()  # [p, j]
        self.init_weight()

    def random_mask_condition(self, condition):
        """
        During model.train(): Random mask the condition.
        During model.eval(): no operation.
        condition: [B, j*2]
        """
        batch_size = condition.shape[0]
        # mask poses
        if self.cond_pose_mask_prob > 0:
            mask = torch.bernoulli(
                torch.ones(batch_size, device=condition.device) * self.cond_pose_mask_prob
            ).view(batch_size, 1)  # 1-> use null condition, 0-> use real condition
            condition = condition * (1. - mask)

        # TODO: mask parts
        if self.cond_part_mask_prob > 0:
            final_mask = np.ones((batch_size, self.n_joints), dtype=np.float32)
            mask = torch.bernoulli(
                torch.ones(batch_size, self.num_parts) * self.cond_part_mask_prob
            ).numpy().astype(bool)  # [b, p]
            for idx, row in enumerate(mask):
                if np.sum(row) > 0:
                    selected_mask = self.part_mask[row]  # [s, j]
                    overlapped_mask = np.prod(selected_mask, axis=0)  # [j]
                    final_mask[idx] = overlapped_mask
            final_mask = torch.tensor(final_mask, device=condition.device).unsqueeze(-1)  # [b, j, 1]
            condition = condition.view(batch_size, self.n_joints, -1) * final_mask
            condition = condition.view(batch_size, -1)

        # mask joints
        if self.cond_joint_mask_prob > 0:
            mask = torch.bernoulli(
                torch.ones((batch_size, self.n_joints, 1), device=condition.device) * self.cond_joint_mask_prob
            ) # 1-> use null condition, 0-> use real condition
            condition = condition.view(batch_size, self.n_joints, -1) * (1. - mask)
            condition = condition.view(batch_size, -1)

        return condition
    def freeze(self):
        for name, param in self.named_parameters():
            if 'copy' in name or 'zc' in name:
                
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        self.infant_cond.requires_grad = True
        # import ipdb;ipdb.set_trace()                    

    def init_weight(self):
        # self.freeze()
        for param in self.parameters():
            param.requires_grad = False
        getattr(self, f'pre_dense_copy').weight.copy_(getattr(self, f'pre_dense').weight)
        getattr(self, f'pre_dense_copy').bias.copy_(getattr(self, f'pre_dense').bias)
        getattr(self, f'pre_dense_t_copy').weight.copy_(getattr(self, f'pre_dense_t').weight)
        getattr(self, f'pre_dense_t_copy').bias.copy_(getattr(self, f'pre_dense_t').bias)
        getattr(self, f'pre_gnorm_copy').weight.copy_(getattr(self, f'pre_gnorm').weight)
        getattr(self, f'pre_gnorm_copy').bias.copy_(getattr(self, f'pre_gnorm').bias)
        for idx in range(self.n_blocks):
            # import ipdb;ipdb.set_trace()
            getattr(self, f'b{idx+1}_dense1_copy').weight.copy_(getattr(self, f'b{idx+1}_dense1').weight)
            getattr(self, f'b{idx+1}_dense1_copy').bias.copy_(getattr(self, f'b{idx+1}_dense1').bias)
            getattr(self, f'b{idx+1}_dense1_t_copy').weight.copy_(getattr(self, f'b{idx+1}_dense1_t').weight)
            getattr(self, f'b{idx+1}_dense1_t_copy').bias.copy_(getattr(self, f'b{idx+1}_dense1_t').bias)
            getattr(self, f'b{idx+1}_gnorm1_copy').weight.copy_(getattr(self, f'b{idx+1}_gnorm1').weight)
            getattr(self, f'b{idx+1}_gnorm1_copy').bias.copy_(getattr(self, f'b{idx+1}_gnorm1').bias)
            
            getattr(self, f'b{idx+1}_dense2_copy').weight.copy_(getattr(self, f'b{idx+1}_dense2').weight)
            getattr(self, f'b{idx+1}_dense2_copy').bias.copy_(getattr(self, f'b{idx+1}_dense2').bias)
            getattr(self, f'b{idx+1}_dense2_t_copy').weight.copy_(getattr(self, f'b{idx+1}_dense2_t').weight)
            getattr(self, f'b{idx+1}_dense2_t_copy').bias.copy_(getattr(self, f'b{idx+1}_dense2_t').bias)
            getattr(self, f'b{idx+1}_gnorm2_copy').weight.copy_(getattr(self, f'b{idx+1}_gnorm2').weight)
            getattr(self, f'b{idx+1}_gnorm2_copy').bias.copy_(getattr(self, f'b{idx+1}_gnorm2').bias)
            
        self.freeze()

    def generate_part_mask(self):
        """
        given parts
        """
        part_list = [[1, 2, 3], [4, 5, 6], [11, 12, 13],
        [14, 15, 16], [0, 7, 8, 9, 10]]
        self.num_parts = len(part_list)

        part_mask = np.ones((self.num_parts, self.n_joints))  # [p, 17]
        for idx, part in enumerate(part_list):
            part_mask[idx][part] = 0

        return part_mask

    def forward(self, batch, t, condition=None):
        """
        batch: [B, j, 3] or [B, j, 1]
        t: [B]
        condition: [B, j, 2 or 3]
        mask: [B, j, 2 or 3] only used during evaluation
        Return: [B, j, 3] or [B, j, 1] same dim as batch
        """
        bs = batch.shape[0]
        

        batch = batch.view(bs, -1)  # [B, j*3]
        c = self.zc_layer_1(self.infant_cond)
        c = self.act(c)
        c = batch+c
        
        if self.time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = self.gauss_proj(torch.log(used_sigmas))
        elif self.time_embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = self.posit_proj(timesteps)
        else:
            raise ValueError(f'time embedding type {self.time_embedding_type} unknown.')

        temb = self.shared_time_embed(temb)

        # cond embedding
        
        c = self.pre_dense_copy(c)
        c+= self.pre_dense_t_copy(temb)
        c0 = self.zc_layer_2(c)
        c = self.pre_gnorm_copy(c)
        c = self.act(c)
        c = self.dropout(c)
        
        h = self.pre_dense(batch)
        h += self.pre_dense_t(temb)
        h += c0
        h = self.pre_gnorm(h)
        h = self.act(h)
        h = self.dropout(h)
        
        
        

        for idx in range(self.n_blocks):
            orc = c 
            c = getattr(self, f'b{idx+1}_dense1_copy')(c)
            c += getattr(self, f'b{idx+1}_dense1_t_copy')(temb)
            c1 = getattr(self, f'zc_b{idx+1}_1')(c)
            
           
         
            c = getattr(self, f'b{idx+1}_gnorm1_copy')(c)
            c = self.act(c)
            c = self.dropout(c)
            
            
            
            c = getattr(self, f'b{idx+1}_dense2_copy')(c)
            c = getattr(self, f'b{idx+1}_dense2_t_copy')(temb)
            c2 = getattr(self, f'zc_b{idx+1}_2')(c)
            
            c = getattr(self, f'b{idx+1}_gnorm2_copy')(c)
            c = self.act(c)
            c = self.dropout(c)
            
           
            c = orc + c
            
            
            h1 = getattr(self, f'b{idx+1}_dense1')(h)
            h1 += getattr(self, f'b{idx+1}_dense1_t')(temb)
            h1 += c1
            h1 = getattr(self, f'b{idx+1}_gnorm1')(h1)
            h1 = self.act(h1)
            h1 = self.dropout(h1)
            

            h2 = getattr(self, f'b{idx+1}_dense2')(h1)
            h2 += getattr(self, f'b{idx+1}_dense2_t')(temb)
            h2 += c2
            h2 = getattr(self, f'b{idx+1}_gnorm2')(h2)
            h2 = self.act(h2)
            h2 = self.dropout(h2)

           
           
            
            

            h = h + h2

        res = self.post_dense(h)  # [B, j*3]
        res = res.view(batch.shape[0], self.n_joints, -1)  # [B, j, 3]

        ''' normalize the output '''
        if self.config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((bs, 1, 1))
            res = res / used_sigmas

        return res
    
    
    
   
        