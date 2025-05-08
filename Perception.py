import torch
import torch.nn as nn
from video_decoder import DecoderVideo
from einops import rearrange
import os
import argparse
import math
import glob
import random
import itertools
import datetime
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.autograd import Variable
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=768,seq_len=64,C=62):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.conv = nn.Conv1d(21,seq_len, 1,1)



    def forward(self, x: Tensor) -> Tensor: #[b,62,400] -> [b,21,768]
        x = x.unsqueeze(1)
        x = self.shallownet(x)
        x = self.projection(x)
        x = self.conv(x)
        return x



class Perception_Reconstruction(nn.Module):
    def __init__(self, h=768, in_dim=4096, out_dim=768, seq_len=64, n_blocks=4, drop=.15, clip_size=768, blurry_recon=True, clip_scale=1):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.blurry_recon = blurry_recon
        self.clip_scale = clip_scale
        # self.mixer_blocks1 = nn.ModuleList([
        #     self.mixer_block1(h, drop) for _ in range(n_blocks)
        # ])
        # self.mixer_blocks2 = nn.ModuleList([
        #     self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        # ])
        self.globalnet = PatchEmbedding(out_dim,seq_len,62)
        
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = PatchEmbedding(out_dim,seq_len,12)
        #[b,62,768*2]
        #self.out = nn.Linear(emb_dim*2, out_dim)
        self.cls=nn.Linear(out_dim*2,out_dim)
        
  
        if self.blurry_recon:
            self.blin1 = nn.Linear(out_dim*seq_len,4*28*28,bias=True)
            self.bdropout = nn.Dropout(.3)
            self.bnorm = nn.GroupNorm(1, 64)
            self.bupsampler = DecoderVideo(
                in_channels=64,
                out_channels=4,
                up_block_types=["AttnUpDecoderBlock2D","AttnUpDecoderBlock2D","AttnUpDecoderBlock2D"],
                block_out_channels=[32, 64, 128],
                layers_per_block=4,
            )
            self.b_maps_projector = nn.Sequential(
                nn.Conv2d(64, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=True),
            )
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    

    def forward(self, x, time):
        # make empty tensors
        b = torch.Tensor([[0.],[0.]])
        
        global_feature = self.globalnet(x) #[b,64,768]
        #global_feature = global_feature.view(x.size(0), -1)
        # global_feature = self.out(global_feature)
        occipital_x = x[:, self.occipital_index, :]
        # print("occipital_x.shape = ", occipital_x.shape)
        occipital_feature = self.occipital_localnet(occipital_x)
        # print("occipital_feature.shape = ", occipital_feature.shape)
        out = torch.cat((global_feature, occipital_feature), -1)
        #out = self.cls(torch.cat((global_feature, occipital_feature), -1))
        out =self.cls(out) #[b,64,768]

            
        x = out.reshape(out.size(0), -1) #[b,64*768]
        

        if self.blurry_recon:
            b = self.blin1(x)
            b = self.bdropout(b)
            b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
            b = self.bnorm(b)
            b_aux = self.b_maps_projector(b).flatten(2).permute(0,2,1)
            b_aux = b_aux.view(len(b_aux), 49, 512)
            y = self.bupsampler(b, time = time)
            b = (y, b_aux)

        return b

class Inception_Extension(nn.Module):
    def __init__(self, h=400, in_dim=400, out_dim=400, expand=6, seq_len=62, n_blocks=1, drop=.15):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.expand = expand
        self.lin0 = self.mlp(in_dim, h, drop)
        self.lin1 = self.mlp(h, out_dim*expand, drop)
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
    
    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )
        
    def forward(self, x):
        # make empty tensors
        b = torch.Tensor([[0.],[0.]])
        
        x = self.lin0(x) 
        # Mixer blocks
        # residual1 = x
        # residual2 = x.permute(0,2,1)
        # for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
        #     x = block1(x) + residual1 
        #     residual1 = x
        #     x = x.permute(0,2,1)
            
        #     x = block2(x) + residual2
        #     residual2 = x
        #     x = x.permute(0,2,1)
            
        
        x = self.lin1(x)
        x = rearrange(x,'b n (d f) -> (b f) n d',f=self.expand)
        return x

if __name__=="__main__":
    model =Perception_Reconstruction()
    x = torch.rand((1,62,400))
    y =model(x)
    print(y.shape)