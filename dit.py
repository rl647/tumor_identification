#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 08:59:48 2024

@author: runfeng
"""

import nibabel as nib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import torch.nn.functional as F
import torch
import torchvision
from torchvision import transforms 
from PIL import Image
from torch.utils.data import DataLoader
#%%

input_data = {}

with open('my.env', 'r') as f:
    lines=f.readlines()
    trainpath=lines[0].strip()
    labelpath=lines[1].strip()
    parampath=lines[2].strip()

path = trainpath
for file in os.listdir(path):
        
    nii_img = nib.load(f"{path}/{file}")
    file = file[:-7]
    input_image = nii_img.get_fdata()
    input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image))# this is to normaliz the input

    input_data[file]=input_image
label_data = {}
import numpy as np
path = labelpath
for file in os.listdir(path):
    nii_img = nib.load(f"{path}/{file}")
    file = file[:-7]
    label_array = nii_img.get_fdata()
    label_array = (label_array - np.min(label_array)) / (np.max(label_array) - np.min(label_array))
    

    label_data[file] = label_array
    



#%%
x_train = []
y_train = []
x_test = []
y_test = []
for key, val in input_data.items():
    if int(key[:2])<43:
        x_train.append(torch.tensor(val, dtype=torch.float32))  
        y_train.append(torch.tensor(label_data[key]))
    else:
        x_test.append(torch.tensor(val, dtype=torch.float32))  
        y_test.append(torch.tensor(label_data[key]))

train_data = []
test_data = []
for i, e in enumerate(x_train):
    train_data.append([e, y_train[i]])
for i, e in enumerate(x_test):
    test_data.append([e, y_test[i]])

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1)
#%%
import torch.nn.functional as F
'''this is from: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=uuckjpW_k1LN'''
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cpu"):

    noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x_0)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x_0)

    noisy_x = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_x.to(device), noise.to(device)

T = 1000
betas = linear_beta_schedule(timesteps=T)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
#%%
import torch
import random
import torchvision.transforms.functional as TF

def random_rotate_flip_tensor(image, label):
    angle = random.uniform(-180, 180)  
    image = TF.rotate(image, angle)
    label = TF.rotate(label, angle)

    if random.random() > 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label)

    if random.random() > 0.5:
        image = TF.vflip(image)
        label = TF.vflip(label)

    return image, label
#%%

import torch
import torch.nn as nn
import math
'''this model is from https://github.com/milmor/diffusion-transformer/blob/main/dit.py'''

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LinformerAttention(nn.Module):
    def __init__(self, seq_len, dim, n_heads, k, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qw = nn.Linear(dim, dim, bias = bias)
        self.kw = nn.Linear(dim, dim, bias = bias)
        self.vw = nn.Linear(dim, dim, bias = bias)

        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))

        self.ow = nn.Linear(dim, dim, bias = bias)

    def forward(self, x):
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        B, L, D = q.shape
        q = torch.reshape(q, [B, L, self.n_heads, -1])
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.reshape(k, [B, L, self.n_heads, -1])
        k = torch.permute(k, [0, 2, 3, 1])
        v = torch.reshape(v, [B, L, self.n_heads, -1])
        v = torch.permute(v, [0, 2, 3, 1])
        k = torch.matmul(k, self.E[:L, :])

        v = torch.matmul(v, self.F[:L, :])
        v = torch.permute(v, [0, 1, 3, 2])

        qk = torch.matmul(q, k) * self.scale
        attn = torch.softmax(qk, dim=-1)

        v_attn = torch.matmul(attn, v)
        v_attn = torch.permute(v_attn, [0, 2, 1, 3])
        v_attn = torch.reshape(v_attn, [B, L, D])

        x = self.ow(v_attn)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TransformerBlock(nn.Module):
    def __init__(self, seq_len, dim, heads, mlp_dim, k, rate=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = LinformerAttention(seq_len, dim, heads, k)
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate),
        )
        self.gamma_1 = nn.Linear(dim, dim)
        self.beta_1 = nn.Linear(dim, dim)
        self.gamma_2 = nn.Linear(dim, dim)
        self.beta_2 = nn.Linear(dim, dim)
        self.scale_1 = nn.Linear(dim, dim)
        self.scale_2 = nn.Linear(dim, dim)

        nn.init.zeros_(self.gamma_1.weight)
        nn.init.zeros_(self.beta_1.weight)
        nn.init.zeros_(self.gamma_1.bias)
        nn.init.zeros_(self.beta_1.bias)  

        nn.init.zeros_(self.gamma_2.weight)
        nn.init.zeros_(self.beta_2.weight)
        nn.init.zeros_(self.gamma_2.bias)
        nn.init.zeros_(self.beta_2.bias)  

        nn.init.zeros_(self.scale_1.weight)
        nn.init.zeros_(self.scale_2.weight)
        nn.init.zeros_(self.scale_1.bias)
        nn.init.zeros_(self.scale_2.bias)  

    def forward(self, x, c):
        #c = self.ln_act(c)
        scale_msa = self.gamma_1(c)
        shift_msa = self.beta_1(c)
        scale_mlp = self.gamma_2(c)
        shift_mlp = self.beta_2(c)
        gate_msa = self.scale_1(c).unsqueeze(1)
        gate_mlp = self.scale_2(c).unsqueeze(1)
        x = self.attn(modulate(self.ln_1(x), shift_msa, scale_msa)) * gate_msa + x
        return self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)) * gate_mlp + x

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        # Zero-out output layers:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)        

    def forward(self, x, c):
        scale = self.gamma(c)
        shift = self.beta(c)
        x = modulate(self.ln_final(x), shift, scale)
        x = self.linear(x)
        return x



class DiT(nn.Module):
    def __init__(self, img_size, dim=64, patch_size=16,
                 depth=3, heads=4, mlp_dim=512, k=64, in_channels=10):
        super(DiT, self).__init__()
        self.dim = dim
        self.n_patches = (img_size // patch_size)**2 
        self.depth = depth
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, dim))
        self.patches = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, 
                      stride=patch_size, padding=0, bias=False),
        )
        
        self.transformer = nn.ModuleList()
        for i in range(self.depth):
            self.transformer.append(
                TransformerBlock(
                    self.n_patches, dim, heads, mlp_dim, k)
            )

        self.emb = nn.Sequential(
            PositionalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.final = FinalLayer(dim, patch_size, in_channels)
        self.ps = nn.PixelShuffle(patch_size)

    def forward(self, x, t):
        t = self.emb(t)
        x = self.patches(x)
        B, C, H, W = x.shape
        x = x.permute([0, 2, 3, 1]).reshape([B, H * W, C])
        x += self.pos_embedding
        for layer in self.transformer:
            x = layer(x, t)

        x = self.final(x, t).permute([0, 2, 1])
        x = x.reshape([B, -1, H, W])
        x = self.ps(x)
        return x

# Instantiate the model with correct number of input channels
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiT(img_size=320).to(device)

#%%
num_epochs=1000
# model.load_state_dict(torch.load(f'.../parameters/dit.pth',map_location=device),strict=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.BCEWithLogitsLoss()


for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        inputs = inputs.permute(0, 3, 2, 1)
        t = torch.randint(0, T, (inputs.size(0),), dtype=torch.long)
        noisy_inputs, _ = forward_diffusion_sample(inputs, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        inputs_transformed, labels_transformed = [], []

        #for img, lbl in zip(noisy_inputs, labels):
        #    img_t, lbl_t = random_rotate_flip_tensor(img, lbl)
        #    inputs_transformed.append(img_t)
        #    labels_transformed.append(lbl_t)
        
        #noisy_inputs = torch.stack(inputs_transformed).to(device)
        #labels = torch.stack(labels_transformed).to(device)
        
        noisy_inputs = noisy_inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(noisy_inputs,t)
        outputs = outputs.permute(0, 3, 2, 1)
        loss = loss_function(outputs, labels.float())
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    torch.save(model.state_dict(), parampath)

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DiT(img_size=320).to(device)
model.load_state_dict(torch.load(parampath,map_location=device),strict=True)



#%%

##plot 

device = "cpu" if torch.cuda.is_available() else "cpu"

model = DiT(img_size=320).to(device)
model.load_state_dict(torch.load(parampath,map_location=device),strict=True)




#%%
inputs = []
predictions = []
actual = []
model.eval()
with torch.no_grad():
    # for input1, keys in train_dataloader: 
    for input1, keys in test_dataloader: 
        inputs.append(input1)

        input1 = input1.permute(0, 3, 1, 2)
        
        input1 = input1.to(device)
        tt = torch.zeros(input1.shape[0], dtype=input1.dtype).to(device)
        predictions.append(model(input1,tt).permute(0, 2, 3, 1).detach().cpu().numpy())
        actual.append(keys.detach().cpu().numpy())
#%%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

num_groups = len(inputs) 
num_slices = 10  

fig, axes = plt.subplots(nrows=3 * num_slices, ncols=num_groups, figsize=(num_groups * 3, num_slices * 9))

for group_idx in range(num_groups):

    input_imgs = np.squeeze(inputs[group_idx].detach().cpu().numpy(),axis=(0)) 
    actual_imgs = np.squeeze(actual[group_idx],axis=(0)) 
    pred_imgs = np.squeeze(predictions[group_idx],axis=(0)) 
    pred_imgs = (pred_imgs >= 0.5).astype(int)

    for slice_idx in range(num_slices):
        input_img = input_imgs[:, :, slice_idx]  
        actual_img = actual_imgs[:, :, slice_idx]  
        pred_img = pred_imgs[:, :, slice_idx]  

        axes[slice_idx * 3, group_idx].imshow(input_img, cmap='gray')
        if group_idx == 0:
            axes[slice_idx * 3, group_idx].set_ylabel('Input', fontsize=12)
        axes[slice_idx * 3, group_idx].axis('off')

        axes[slice_idx * 3 + 1, group_idx].imshow(actual_img, cmap='gray')
        if group_idx == 0:
            axes[slice_idx * 3 + 1, group_idx].set_ylabel('Actual', fontsize=12)
        axes[slice_idx * 3 + 1, group_idx].axis('off')

        axes[slice_idx * 3 + 2, group_idx].imshow(pred_img, cmap='gray')
        if group_idx == 0:
            axes[slice_idx * 3 + 2, group_idx].set_ylabel('Predicted', fontsize=12)
        axes[slice_idx * 3 + 2, group_idx].axis('off')

plt.tight_layout()
# plt.savefig('train.svg', bbox_inches='tight') 
plt.savefig('test.png', bbox_inches='tight')

#plt.show()







































