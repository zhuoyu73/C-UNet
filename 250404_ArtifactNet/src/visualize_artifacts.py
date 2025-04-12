# visualize_artifacts.py
# ZS wrote this script to visualize the artifact map and compare results with the best model

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from scipy.io import loadmat
from torch.nn import functional as F
from .models.artifactnet import ArtifactNet  
import argparse
def load_slice(data, b, s, t):
    """Get a single complex slice from the 5D MRE data [H, W, B, S, T]"""
    slice_complex = data[:, :, b, s, t]
    return slice_complex
def tensor_to_numpy(t):
    return t.detach().cpu().numpy()
def visualize(clean_path, lowrank_path, model_path, b, s, t): #ZS example
    # Load data
    clean = loadmat(clean_path)['img']
    clean = np.squeeze(clean)  # [120,120,4,16,24]
    lowrank = loadmat(lowrank_path)['img']
    print(lowrank.shape)
    lowrank = np.squeeze(lowrank)
    print(lowrank.shape)
    clean_slice = clean[:,:,b,s,t]
    lowrank_slice = lowrank[:,:,b,s,t]

    lowrank_slice = np.stack([np.real(lowrank_slice), np.imag(lowrank_slice)], axis=0)
    clean_slice = np.stack([np.real(clean_slice), np.imag(clean_slice)], axis=0)
    # ZS Convert to tensor before padding
    lowrank_slice = torch.from_numpy(lowrank_slice).float()
    clean_slice = torch.from_numpy(clean_slice).float()
    # ZS Pad to 128x128 
    lowrank_slice = F.pad(lowrank_slice, (4,4,4,4), mode="constant", value=0)
    clean_slice = F.pad(clean_slice, (4,4,4,4), mode="constant", value=0)
    artifact_true = lowrank_slice - clean_slice
    clean_slice = clean_slice[0].numpy() + 1j * clean_slice[1].numpy()
    artifact_true = artifact_true[0].numpy() + 1j * artifact_true[1].numpy()


    #print(lowrank_slice.shape)
    #print(artifact_true.shape)
    #input_tensor = np.stack([np.real(lowrank_slice), np.imag(lowrank_slice)], axis=0)
    input_tensor = lowrank_slice.unsqueeze(0).float().to('cuda')  # [1, 2, 120, 120]    
    
    # Load model
    model = ArtifactNet().to('cuda')
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()
    # Predict artifact
    with torch.no_grad():
        pred = model(input_tensor).cpu().squeeze(0)  # [2, H, W]

    print("Input Tensor:", input_tensor.shape, input_tensor.dtype)
    print("Pred Tensor:", pred.shape)
    print("pred[0].mean():", pred[0].mean().item())
    print("pred[1].mean():", pred[1].mean().item())

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(pred[0], cmap='gray')
    axes[0].set_title("Real")
    axes[1].imshow(pred[1], cmap='gray')
    axes[1].set_title("Imag")
    plt.show()


    lowrank_slice = lowrank_slice[0].numpy() + 1j * lowrank_slice[1].numpy()

    artifact_pred = pred[0].numpy() + 1j * pred[1].numpy()
    # Reconstructed (denoised) image
    corrected = lowrank_slice - artifact_pred
    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    images = [
        np.abs(clean_slice),           # clean image (fully rank)
        np.abs(lowrank_slice),         # input (low-rank)
        np.abs(artifact_pred),         # predicted artifact map
        np.abs(artifact_true),         # ground truth artifact map
        np.abs(corrected)              # denoised image
    ]
    titles = ['Clean (fullyrank)','Input (Lowrank)', 'Predicted Artifact', 'True Artifact', 'Denoised']
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=str, required=True, help='Path to fullysampled img.mat')
    parser.add_argument('--lowrank', type=str, required=True, help='Path to lowrank img.mat')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pt file')
    parser.add_argument('--b', type=int, default=3)
    parser.add_argument('--s', type=int, default=12)
    parser.add_argument('--t', type=int, default=23)
    args = parser.parse_args()
    visualize(args.clean, args.lowrank, args.model, args.b, args.s, args.t)