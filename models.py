"""
Different EEG encoders for comparison

SA GA

shallownet, deepnet, eegnet, conformer, tsconv
"""


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


# class PatchEmbedding(nn.Module):
#     def __init__(self, emb_size=40):
#         # self.patch_size = patch_size
#         super().__init__()
#         self.tsconv = nn.Sequential(
#             nn.Conv2d(1, 40, (1, 25), (1, 1)),
#             nn.AvgPool2d((1, 51), (1, 5)),
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             nn.Conv2d(40, 40, (63, 1), (1, 1)),
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             nn.Dropout(0.5),
#         )
#         self.deepnet = nn.Sequential(
#             nn.Conv2d(1, 25, (1, 10), (1, 1)),
#             nn.Conv2d(25, 25, (63, 1), (1, 1)),
#             nn.BatchNorm2d(25),
#             nn.ELU(),
#             nn.MaxPool2d((1, 2), (1, 2)),
#             nn.Dropout(0.5),

#             nn.Conv2d(25, 50, (1, 10), (1, 1)),
#             nn.BatchNorm2d(50),
#             nn.ELU(),
#             nn.MaxPool2d((1, 2), (1, 2)),
#             nn.Dropout(0.5),

#             nn.Conv2d(50, 100, (1, 10), (1, 1)),
#             nn.BatchNorm2d(100),
#             nn.ELU(),
#             nn.MaxPool2d((1, 2), (1, 2)),
#             nn.Dropout(0.5),

#             nn.Conv2d(100, 200, (1, 10), (1, 1)),
#             nn.BatchNorm2d(200),
#             nn.ELU(),
#             nn.MaxPool2d((1, 2), (1, 2)),
#             nn.Dropout(0.5),

#         )

#         self.eegnet = nn.Sequential(
#             nn.Conv2d(1, 8, (1, 64), (1, 1)),
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 16, (63, 1), (1, 1)),
#             nn.BatchNorm2d(16),
#             nn.ELU(),
#             nn.AvgPool2d((1, 2), (1, 2)),
#             nn.Dropout(0.5),
#             nn.Conv2d(16, 16, (1, 16), (1, 1)),
#             nn.BatchNorm2d(16), 
#             nn.ELU(),
#             # nn.AvgPool2d((1, 2), (1, 2)),
#             nn.Dropout2d(0.5)
#         )

#         self.shallownet = nn.Sequential(
#             nn.Conv2d(1, 40, (1, 25), (1, 1)),
#             nn.Conv2d(40, 40, (63, 1), (1, 1)),
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             nn.AvgPool2d((1, 51), (1, 5)),
#             nn.Dropout(0.5),
#         )

#         self.projection = nn.Sequential(
#             nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
#             Rearrange('b e (h) (w) -> b (h w) e'),
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.tsconv(x)
#         return x

class shallownet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(shallownet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(1040*(T//200), out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
    
class deepnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(deepnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (C, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(800*(T//200), out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
    
class eegnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(eegnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5)
        )
        self.out = nn.Linear(416*(T//200), out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

class tsconv(nn.Module):
    def __init__(self, out_dim, C, T):
        super(tsconv, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(1040*(T//200), out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=768,seq_len=64,C=62):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.GELU(),
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


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor):
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=8,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):  
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, out_dim):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(280, out_dim),
        )

    def forward(self, x):
        # x = x.contiguous().view(x.size(0), -1)
        out = self.clshead(x)
        return out


class conformer(nn.Module):
    def __init__(self, emb_size=768, depth=3, seq_len=64,out_dim=40):
        super().__init__()
        # 初始化PatchEmbedding层
        self.patch = PatchEmbedding(emb_size,seq_len)
        # 初始化TransformerEncoder层
        self.transformer = TransformerEncoder(depth, emb_size)
        # 初始化ClassificationHead层
        # nn.Linear(280, out_dim)
        self.cls=ClassificationHead(emb_size, out_dim)
    
    def forward(self,x):
        # 将输入x通过PatchEmbedding层
        x = self.patch(x)
        # 将PatchEmbedding层的输出通过TransformerEncoder层
        x = self.transformer(x)
        # 将TransformerEncoder层的输出通过ClassificationHead层
        x = self.cls(x)
        # 返回ClassificationHead层的输出
        return x
        
class conformer_wo_cls(nn.Module):
    def __init__(self, emb_size=768, depth=3, seq_len=64,out_dim=40,C=62):
        super().__init__()
        # 初始化PatchEmbedding层
        self.patch = PatchEmbedding(emb_size,seq_len,C=C)
        # 初始化TransformerEncoder层
        self.transformer = TransformerEncoder(depth, emb_size)
        # 初始化ClassificationHead层
        # nn.Linear(280, out_dim)
        #self.cls=ClassificationHead(emb_size, out_dim)
    
    def forward(self,x):
        # 将输入x通过PatchEmbedding层
        x = self.patch(x)
        # 将PatchEmbedding层的输出通过TransformerEncoder层
        x = self.transformer(x)
        # 将TransformerEncoder层的输出通过ClassificationHead层
        
        # 返回ClassificationHead层的输出
        return x



class TemporalConv(nn.Module):
    """ EEG to Patch Embedding"""
    
    def __init__(self,in_chans=1,out_chans=8,in_dim=5,eeg_chans=62,seq_len=256,emb_dim=1664):
        super().__init__()
        if in_dim==400:
            self.conv1 = nn.Conv2d(in_chans,out_chans,kernel_size=(1,15),stride=(1,8),padding=(0,7))
        else:
            self.conv1 = nn.Conv2d(in_chans,out_chans,kernel_size=(1,1))
        self.gelu1=nn.GELU()
        self.norm1 = nn.GroupNorm(4,out_chans)
        self.drop1 = nn.Dropout(0.5)
        if in_dim ==400:
            self.conv2 = nn.Conv2d(out_chans,out_chans,kernel_size=(1,3),padding=(0, 1))
        else:
            self.conv2 = nn.Conv2d(out_chans,out_chans,kernel_size=(1,1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4,out_chans)
       
        if in_dim ==400:
            self.conv3 = nn.Conv2d(out_chans,out_chans,kernel_size=(1,3),padding=(0, 1))
        else:
            self.conv3 = nn.Conv2d(out_chans,out_chans,kernel_size=(1,1))
        self.norm3 = nn.GroupNorm(4,out_chans)
        self.gelu3 = nn.GELU()
        if in_dim ==400:
            self.linear = nn.Linear(in_dim,emb_dim)
        else:
            self.linear = nn.Linear(in_dim*out_chans,emb_dim)
        self.norm4 = nn.LayerNorm(emb_dim)
        self.gelu4 = nn.GELU()
       
        self.linear2 = nn.Linear(eeg_chans,seq_len)
        self.norm5 = nn.GroupNorm(4,out_chans)
        self.gelu5 = nn.GELU()
       
    def forward(self,x,**kwargs):
        # x = rearrange(x,'B N A T -> B (N A) T')
        B, NA, T= x.shape 
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        
        x = rearrange(x,'b c n t -> b c t n')
        x = self.gelu5(self.norm5(self.linear2(x)))
        x = rearrange(x,'B C t n -> B n (t C)')
        x = self.gelu4(self.norm4(self.linear(x)))
        x = self.drop1(x)
        
        return x 



class glfnet(nn.Module):
    def __init__(self, out_dim=40, emb_dim=768, seq_len=64,depth=1):
        super(glfnet, self).__init__()
        
        self.globalnet = TemporalConv(in_dim=5)
        
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = TemporalConv(in_dim=5,eeg_chans=12)
        
        #[b,62,768*2]
        #self.out = nn.Linear(emb_dim*2, out_dim)
        self.cls=nn.Linear(emb_dim*2,emb_dim)
        
        self.trans = TransformerEncoder(depth,emb_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):               #input:(batch,1,C,T)
        global_feature = self.globalnet(x) #[b,64,768]
        #global_feature = global_feature.view(x.size(0), -1)
        # global_feature = self.out(global_feature)
        occipital_x = x[:, self.occipital_index, :]
        # print("occipital_x.shape = ", occipital_x.shape)
        occipital_feature = self.occipital_localnet(occipital_x)
        # print("occipital_feature.shape = ", occipital_feature.shape)
        out = torch.cat((global_feature, occipital_feature), -1)
        #out = self.cls(torch.cat((global_feature, occipital_feature), -1))
        out =self.cls(out)
        out = self.dropout(self.trans(out))
        return out

class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim):
        super(mlpnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, out_dim)
        )
        
    def forward(self, x):               #input:(batch,C,5)
        out = self.net(x)
        return out

class glfnet_mlp(nn.Module):
    def __init__(self, out_dim=40, emb_dim=768, seq_len=64,depth=1):
        super(glfnet_mlp, self).__init__()
        
        self.globalnet = mlpnet(out_dim = emb_dim,input_dim=62*5)
        
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = mlpnet(out_dim = emb_dim,input_dim=12*5)
        
        self.cls=nn.Linear(emb_dim*2,emb_dim)
        
        self.trans = TransformerEncoder(depth,emb_dim)
        self.dropout = nn.Dropout(0.5)
        
    
    def forward(self, x):               #input:(batch,C,5)
        global_feature = self.globalnet()
        # global_feature = global_feature.view(x.size(0), -1)
        # global_feature = self.out(global_feature)
        occipital_x = x[:, self.occipital_index, :]
        # print("occipital_x.shape = ", occipital_x.shape)
        occipital_feature = self.occipital_localnet(occipital_x)
        # print("occipital_feature.shape = ", occipital_feature.shape)
        out = self.cls(torch.cat((global_feature, occipital_feature), -1))
        out = self.dropout(self.trans(out))
      
        return out


class Semantic_Reconstruction(nn.Module):
    def __init__(self, h=400, in_dim=13447, out_dim=40, seq_len=62, n_blocks=3, drop=.15, clip_size=768, blurry_recon=False, clip_scale=1):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.blurry_recon = blurry_recon
        self.clip_scale = clip_scale
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
        
        
            
    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
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
        
        
        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)
            
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)
            
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x) #[b,64,768]
    
        return backbone


class eeg2video(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x 




if __name__ == "__main__":
    # model = glfnet(out_dim=3, emb_dim=256, C=62, T=200)  #input:(batch,channels_conv,channels_eeg,data_num)  output:(batch,num_classes)
    # model = eeg2video()
    # model.backbone = glfnet(emb_dim=1664,seq_len=256)
    # model.proj = ClassificationHead(768,40)
    # x = torch.rand(size=(1, 62, 400))
    # print(x.shape)
    # y = model.backbone(x)
    # print(y.shape)
    # y=model.proj(y)
    # print(y.shape)  #if input(b,1,1,3000),then the output is(1,num_classes)
    model = TemporalConv()
    x = torch.rand(size=(1, 62, 400))
    print(model(x).shape)
    