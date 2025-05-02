#pad_test
    
import os
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


    
lowrank = loadmat('/mnt/zhuoyu1/zhuoyu/fully+osci/1702_AP/1702_AP_B1000/mbmre_both/img.mat')['img']
clean = loadmat('/mnt/zhuoyu1/zhuoyu/fully+osci/1702_AP/1702_AP_B1000/mbmre/img.mat')['img']

lowrank = np.squeeze(lowrank)  # [120, 120, 4, 16, 24]
clean = np.squeeze(clean)

slice_lowrank = lowrank[:, :, 2, 2, 2].astype(np.complex64)
slice_clean = clean[:, :, 2, 2, 2].astype(np.complex64)
slice_artifact = slice_lowrank - slice_clean #ZS
slice_original_artifact = slice_artifact

scale = 1000000
slice_lowrank = slice_lowrank * scale
slice_artifact = slice_artifact * scale
slice_scaled_artifact = slice_artifact

slice_real_artifact = np.real(slice_artifact)
slice_imag_artifact = np.imag(slice_artifact)

slice_lowrank = np.stack([np.real(slice_lowrank), np.imag(slice_lowrank)], axis=0)
slice_artifact = np.stack([np.real(slice_artifact), np.imag(slice_artifact)], axis=0)
# ZS Convert to tensor before padding
slice_lowrank = torch.from_numpy(slice_lowrank).float()
slice_artifact = torch.from_numpy(slice_artifact).float()
# ZS Pad to 128x128 
slice_lowrank = F.pad(slice_lowrank, (4,4,4,4), mode="constant", value=0)
slice_artifact = F.pad(slice_artifact, (4,4,4,4), mode="constant", value=0)

slice_pad_real_artifact = slice_artifact[0].numpy()
slice_pad_imag_artifact = slice_artifact[1].numpy()


fig, axs = plt.subplots(4, 2, figsize=(12, 10))


data00 = np.real(slice_original_artifact)
im00 = axs[0, 0].imshow(data00, cmap='gray', vmin=np.min(data00), vmax=np.max(data00))
axs[0, 0].set_title('Original real artifact')
fig.colorbar(im00, ax=axs[0, 0])

data01 = np.imag(slice_original_artifact)
im01 = axs[0, 1].imshow(data01, cmap='gray', vmin=np.min(data01), vmax=np.max(data01))
axs[0, 1].set_title('Original imag artifact')
fig.colorbar(im01, ax=axs[0, 1])

data10 = np.real(slice_scaled_artifact)
im10 = axs[1, 0].imshow(data10, cmap='gray', vmin=np.min(data10), vmax=np.max(data10))
axs[1, 0].set_title('Scaled real artifact')
fig.colorbar(im10, ax=axs[1, 0])

data11 = np.imag(slice_scaled_artifact)
im11 = axs[1, 1].imshow(data11, cmap='gray', vmin=np.min(data11), vmax=np.max(data11))
axs[1, 1].set_title('Scaled imag artifact')
fig.colorbar(im11, ax=axs[1, 1])

data20 = slice_real_artifact
im20 = axs[2, 0].imshow(data20, cmap='gray', vmin=np.min(data20), vmax=np.max(data20))
axs[2, 0].set_title('Extracted real part')
fig.colorbar(im20, ax=axs[2, 0])

data21 = slice_imag_artifact
im21 = axs[2, 1].imshow(data21, cmap='gray', vmin=np.min(data21), vmax=np.max(data21))
axs[2, 1].set_title('Extracted imag part')
fig.colorbar(im21, ax=axs[2, 1])

data30 = slice_pad_real_artifact
im30 = axs[3, 0].imshow(data30, cmap='gray', vmin=np.min(data30), vmax=np.max(data30))
axs[3, 0].set_title('Padded real part')
fig.colorbar(im30, ax=axs[3, 0])

data31 = slice_pad_imag_artifact
im31 = axs[3, 1].imshow(data31, cmap='gray', vmin=np.min(data31), vmax=np.max(data31))
axs[3, 1].set_title('Padded imag part')
fig.colorbar(im31, ax=axs[3, 1])

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
