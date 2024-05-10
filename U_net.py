#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:39:05 2024

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
path = '.../data/catalyst_open_innovation_challenge/train/compressed_files'
for file in os.listdir(path):
        
    nii_img = nib.load(f"{path}/{file}")
    file = file[:-7]
    input_image = nii_img.get_fdata()
    input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image))# this is to normaliz the input

    input_data[file]=input_image
label_data = {}
import numpy as np
path = '.../astra_zenca/data/output/label1'
for file in os.listdir(path):
    im_frame = Image.open(f'{path}/{file}')
    label_array = np.array(im_frame).astype(np.float32) 
    label_array /= 255
    label_array = label_array[np.newaxis, :, :]  

    label_data[file[:-10]] = label_array
    



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

train_dataloader = DataLoader(train_data, batch_size=12, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1)

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, In_channel,Med_channel,Out_channel,downsample=False):
        super(ResBlock, self).__init__()
        self.stride = 2 if downsample else 1

        self.layer = torch.nn.Sequential(
                    torch.nn.Conv2d(In_channel, Med_channel, 3,stride=self.stride,padding=1),
                    torch.nn.BatchNorm2d(Med_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(Med_channel, Med_channel, 3,stride=1,padding=1),
                    torch.nn.BatchNorm2d(Med_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(Med_channel, Out_channel, 3,stride=1,padding=1),
                    torch.nn.BatchNorm2d(Out_channel),
                    torch.nn.ReLU(),
                )
        if In_channel != Out_channel or self.stride != 1:
            self.res_layer = nn.Conv2d(In_channel, Out_channel, 1, stride=self.stride)
        else:
            self.res_layer = nn.Identity()

    def forward(self, x):
        residual = self.res_layer(x)
        x = self.layer(x)
        return x + residual


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.encoder1 = torch.nn.Sequential(
            ResBlock(10, 32 ,64,False),
            # nn.MaxPool2d(2),
            ResBlock(64, 32 ,64,False),
            # nn.MaxPool2d(2),
            ResBlock(64, 32 ,64,False),
            nn.MaxPool2d(2)
            )
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = torch.nn.Sequential(
            ResBlock(64, 96 ,128,True),
            # nn.MaxPool2d(2),
            ResBlock(128, 96 ,128,False),
            # nn.MaxPool2d(2),
            ResBlock(128, 96 ,128,False),
            nn.MaxPool2d(2)
            )
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = torch.nn.Sequential(
            ResBlock(128, 192, 256,True),
            # nn.MaxPool2d(2),
            ResBlock(256, 192, 256,False),
            # nn.MaxPool2d(2),
            ResBlock(256, 192, 256,False),
            nn.MaxPool2d(2)
            )
        
        self.encoder4 = torch.nn.Sequential(
            ResBlock(256, 384, 512,True),
            ResBlock(512, 384, 512,False),
            ResBlock(512, 384, 512,False)
            
            )
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = ResBlock(512, 256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = ResBlock(256, 128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = ResBlock(128, 64, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # print(x.shape)
        x1 = x
        enc1 = self.encoder1(x)
        # print(enc1.shape)
        # x = self.pool1(enc1)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        # print(enc2.shape)
        # x = self.pool2(enc2)
        # print(x.shape)
        x = self.encoder4(enc3)
        # print(x.shape)
        x = self.upconv3(x)
        x = self.decoder3(torch.cat([x, F.interpolate(enc3, x.size()[2:])], dim=1))

        
        x = self.upconv2(x)
        # print(x.shape)
        x = self.decoder2(torch.cat([x, F.interpolate(enc2, x.size()[2:])], dim=1))
        x = self.upconv1(x)
        x = self.decoder1(torch.cat([x, F.interpolate(enc1, x.size()[2:])], dim=1)) 

        x = F.interpolate(x, size=(x1.size(2) , x1.size(3) ), mode='bilinear', align_corners=True)  # Dynamic upsampling
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

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

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs=10000
model = UNet().to(device)
model.load_state_dict(torch.load(f'/home/runfeng/Dropbox/astra_zenca/code/parameters/Unet_model2.pth',map_location=device),strict=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss_function = nn.BCELoss()


for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        inputs = inputs.permute(0, 3, 1, 2)
        t = torch.randint(0, T, (inputs.size(0),), dtype=torch.long)
        noisy_inputs, _ = forward_diffusion_sample(inputs, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        inputs_transformed, labels_transformed = [], []

        for img, lbl in zip(noisy_inputs, labels):
            img_t, lbl_t = random_rotate_flip_tensor(img, lbl)
            inputs_transformed.append(img_t)
            labels_transformed.append(lbl_t)
        
        noisy_inputs = torch.stack(inputs_transformed).to(device)
        labels = torch.stack(labels_transformed).to(device)
        
        noisy_inputs = noisy_inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_inputs)
        loss = loss_function(outputs, labels.float())
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    torch.save(model.state_dict(), '.../parameters/Unet_model2.pth')

#%%
device = "cpu" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
model.load_state_dict(torch.load(f'.../parameters/Unet_model2.pth',map_location=device),strict=True)


#%%
predictions = []
actual = []
model.eval()
with torch.no_grad():
    # for inputs, keys in train_dataloader: 
    for inputs, keys in test_dataloader: 

        inputs = inputs.permute(0, 3, 1, 2)

        inputs = inputs.to(device)
        predictions.append(model(inputs).detach().cpu().numpy())
        actual.append(keys.detach().cpu().numpy())
#%%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
def calculate_metrics(predictions, actual):
    predictions = predictions.flatten()
    actual = actual.flatten()

    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions, zero_division=0)
    recall = recall_score(actual, predictions)
    f1 = f1_score(actual, predictions)

    cm = confusion_matrix(actual, predictions)
    iou = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0])  

    return accuracy, precision, recall, f1, iou

num_items = len(predictions)
num_rows = int(np.ceil(num_items / 2))  

fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))  

for i, (pred, act) in enumerate(zip(predictions, actual)):
    pred = np.squeeze(pred, axis=(0, 1)) 
    pred = (pred >= 0.5).astype(int)
    act = np.squeeze(act, axis=(0, 1))
    accuracy, precision, recall, f1, iou = calculate_metrics(pred, act)

    ax_pred = axes[i // 2, 2 * (i % 2)]
    ax_pred.imshow(pred)
    ax_pred.set_title(f'Precision: {precision:.3f}, Sensitivity: {recall:.3f}, F1: {f1:.3f}, IOU: {iou:.3f}', fontsize=15)
    ax_pred.axis('off')

    ax_act = axes[i // 2, 2 * (i % 2) + 1]
    ax_act.imshow(act)
    ax_act.set_title('Actual', fontsize=15)
    ax_act.axis('off')
total_plots = num_items * 2
for j in range(total_plots, num_rows * 4):  
    ax = axes.flat[j]
    ax.axis('off')
plt.tight_layout() 
# plt.savefig('/home/runfeng/Dropbox/astra_zenca/data/test.svg', bbox_inches='tight') 
plt.show()


#%%









































