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
    lowrank = np.squeeze(lowrank)
    
    clean_slice = clean[:,:,b,s,t].astype(np.complex64)
    lowrank_slice = lowrank[:,:,b,s,t].astype(np.complex64)
    artifact_slice = lowrank_slice - clean_slice

    scale = 1000000
    clean_slice = clean_slice * scale
    lowrank_slice = lowrank_slice * scale
    artifact_slice = artifact_slice * scale

    clean_slice = np.stack([np.real(clean_slice), np.imag(clean_slice)], axis=0)
    lowrank_slice = np.stack([np.real(lowrank_slice), np.imag(lowrank_slice)], axis=0)
    artifact_slice = np.stack([np.real(artifact_slice), np.imag(artifact_slice)], axis=0)
    # ZS Convert to tensor before padding
    clean_slice = torch.from_numpy(clean_slice).float()
    lowrank_slice = torch.from_numpy(lowrank_slice).float()
    artifact_slice = torch.from_numpy(artifact_slice).float()
    # ZS Pad to 128x128 
    clean_slice = F.pad(clean_slice, (4,4,4,4), mode="constant", value=0)
    lowrank_slice = F.pad(lowrank_slice, (4,4,4,4), mode="constant", value=0)
    artifact_slice = F.pad(artifact_slice, (4,4,4,4), mode="constant", value=0)

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

    #lowrank_slice = lowrank_slice[0].numpy() + 1j * lowrank_slice[1].numpy()
    #artifact_pred = pred[0].numpy() + 1j * pred[1].numpy()
    clean_slice = clean_slice / 1000000
    lowrank_slice = lowrank_slice / 1000000
    artifact_true = artifact_slice / 1000000
    artifact_pred = pred / 1000000
    # Reconstructed (denoised) image
    corrected = lowrank_slice - artifact_pred
    # Plot
    fig, axs = plt.subplots(2, 5, figsize=(18, 4))
    data00 = clean_slice[0].numpy()
    im00 = axs[0, 0].imshow(data00, cmap='gray', vmin=np.min(data00), vmax=np.max(data00))
    axs[0, 0].set_title('Original real clean')
    fig.colorbar(im00, ax=axs[0, 0])

    data10 = clean_slice[1].numpy()
    im10 = axs[1, 0].imshow(data10, cmap='gray', vmin=np.min(data10), vmax=np.max(data10))
    axs[1, 0].set_title('Original imag clean')
    fig.colorbar(im10, ax=axs[1, 0])

    data01 = lowrank_slice[0].numpy()
    im01 = axs[0, 1].imshow(data01, cmap='gray', vmin=np.min(data01), vmax=np.max(data01))
    axs[0, 1].set_title('Original real lowrank')
    fig.colorbar(im01, ax=axs[0, 1])

    data11 = lowrank_slice[1].numpy()
    im11 = axs[1, 1].imshow(data11, cmap='gray', vmin=np.min(data11), vmax=np.max(data11))
    axs[1, 1].set_title('Original imag lowrank')
    fig.colorbar(im11, ax=axs[1, 1])

    data02 = artifact_true[0].numpy()
    im02 = axs[0, 2].imshow(data02, cmap='gray', vmin=np.min(data02), vmax=np.max(data02))
    axs[0, 2].set_title('True real artifact')
    fig.colorbar(im02, ax=axs[0, 2])

    data12 = artifact_true[1].numpy()
    im12 = axs[1, 2].imshow(data12, cmap='gray', vmin=np.min(data12), vmax=np.max(data12))
    axs[1, 2].set_title('True imag artifact')
    fig.colorbar(im12, ax=axs[1, 2])

    data03 = artifact_pred[0].numpy()
    im03 = axs[0, 3].imshow(data03, cmap='gray', vmin=np.min(data03), vmax=np.max(data03))
    axs[0, 3].set_title('Predicted real artifact')
    fig.colorbar(im03, ax=axs[0, 3])

    data13 = artifact_pred[1].numpy()
    im13 = axs[1, 3].imshow(data13, cmap='gray', vmin=np.min(data13), vmax=np.max(data13))
    axs[1, 3].set_title('Predicted imag artifact')
    fig.colorbar(im13, ax=axs[1, 3])

    data04 = corrected[0].numpy()
    im04 = axs[0, 4].imshow(data04, cmap='gray', vmin=np.min(data04), vmax=np.max(data04))
    axs[0, 4].set_title('Denoised real clean')
    fig.colorbar(im04, ax=axs[0, 4])

    data14 = corrected[1].numpy()
    im14 = axs[1, 4].imshow(data14, cmap='gray', vmin=np.min(data14), vmax=np.max(data14))
    axs[1, 4].set_title('Denoised imag clean')
    fig.colorbar(im14, ax=axs[1, 4])

    for ax in axs.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # metrics
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from sklearn.metrics import mean_squared_error

    clean_complex = clean_slice[0].numpy() + 1j * clean_slice[1].numpy()
    corrected_complex = corrected[0].numpy() + 1j * corrected[1].numpy()
    true_artifact_complex = artifact_true[0].numpy() + 1j * artifact_true[1].numpy()
    pred_artifact_complex = artifact_pred[0].numpy() + 1j * artifact_pred[1].numpy()

    # MSE of predicted artifact image and ground truth artifact image
    artifact_mse_val = np.mean(np.abs(true_artifact_complex - pred_artifact_complex) ** 2)
    # PSNR of predicted artifact image and ground truth artifact image
    artifact_psnr_val = psnr(np.abs(true_artifact_complex), np.abs(pred_artifact_complex), data_range=np.max(np.abs(true_artifact_complex)) - np.min(np.abs(true_artifact_complex)))
    artifact_psnr_phase = psnr(np.angle(true_artifact_complex), np.angle(pred_artifact_complex), data_range=np.max(np.angle(true_artifact_complex)) - np.min(np.angle(true_artifact_complex)))

    # MSE of corrected image and clean image
    mse_val = np.mean(np.abs(clean_complex - corrected_complex) ** 2)
    # PSNR of corrected image and clean image
    psnr_val = psnr(np.abs(clean_complex), np.abs(corrected_complex), data_range=np.max(np.abs(clean_complex)) - np.min(np.abs(clean_complex)))
    psnr_phase = psnr(np.angle(clean_complex), np.angle(corrected_complex), data_range=np.max(np.angle(clean_complex)) - np.min(np.angle(clean_complex)))


    print(f"\n[Evaluation Metrics]")
    print(f"MSE between true and predicted artifact image: {artifact_mse_val:.6e}")
    print(f"PSNR (magnitude): {artifact_psnr_val:.2f} dB")
    print(f"PSNR (phase): {artifact_psnr_phase:.2f} dB")
    print(f"MSE between clean and denoised image: {mse_val:.6e}")
    print(f"PSNR (magnitude): {psnr_val:.2f} dB")
    print(f"PSNR (phase): {psnr_phase:.2f} dB")


def find_best_slice(clean, lowrank, model):
    min_mse = float('inf')
    best_idx = (0, 0, 0)
    
    B, S, T = clean.shape[2], clean.shape[3], clean.shape[4]

    for b in range(B):
        for s in range(S):
            for t in range(T):
                # Same slice preprocessing steps
                clean_slice = clean[:,:,b,s,t].astype(np.complex64)
                lowrank_slice = lowrank[:,:,b,s,t].astype(np.complex64)
                artifact_slice = lowrank_slice - clean_slice

                # Amplify
                scale = 1e6
                clean_slice *= scale
                lowrank_slice *= scale
                artifact_slice *= scale

                # Convert to tensor and pad
                clean_slice_t = F.pad(torch.from_numpy(np.stack([np.real(clean_slice), np.imag(clean_slice)])).float(), (4,4,4,4))
                lowrank_slice_t = F.pad(torch.from_numpy(np.stack([np.real(lowrank_slice), np.imag(lowrank_slice)])).float(), (4,4,4,4))
                
                input_tensor = lowrank_slice_t.unsqueeze(0).to('cuda')

                # Predict
                with torch.no_grad():
                    pred = model(input_tensor).cpu().squeeze(0) / scale

                corrected = lowrank_slice_t / scale - pred
                clean_slice_t = clean_slice_t / scale

                # Calculate MSE
                clean_complex = clean_slice_t[0].numpy() + 1j * clean_slice_t[1].numpy()
                corrected_complex = corrected[0].numpy() + 1j * corrected[1].numpy()
                mse = np.mean(np.abs(clean_complex - corrected_complex) ** 2)
                print(f"current slice: b={b}, s={s}, t={t}, mse={mse}")

                if mse < min_mse:
                    min_mse = mse
                    best_idx = (b, s, t)

    return best_idx



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=str, required=True, help='Path to fullysampled img.mat')
    parser.add_argument('--lowrank', type=str, required=True, help='Path to lowrank img.mat')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pt file')
    #parser.add_argument('--b', type=int, default=3)
    #parser.add_argument('--s', type=int, default=12)
    #parser.add_argument('--t', type=int, default=23)
    args = parser.parse_args()
    #visualize(args.clean, args.lowrank, args.model, args.b, args.s, args.t)


    clean_mat = loadmat(args.clean)['img']
    lowrank_mat = loadmat(args.lowrank)['img']
    clean_mat = np.squeeze(clean_mat)
    lowrank_mat = np.squeeze(lowrank_mat)
    model = ArtifactNet().to('cuda')
    model.load_state_dict(torch.load(args.model, map_location='cuda'))
    model.eval()
    b_best, s_best, t_best = find_best_slice(clean_mat, lowrank_mat, model)
    print(f"Best slice (min MSE): b={b_best}, s={s_best}, t={t_best}")
    visualize(args.clean, args.lowrank, args.model, b_best, s_best, t_best)