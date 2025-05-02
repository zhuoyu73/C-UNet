### artifactnet datasets
# ZS

# ZS made the changes below: 
# Data Convension: 
#    a. All the data should go under directory /mnt/external/zhuoyu/fully+osci
#    b. The subject folder names should be put under one of the three .txt files without repeatable use
#    c. The mbmre and oscillate folders should be put under   /subject_id_B1000
#    d. The fully-sampled  img file should be img.mat under mbmre, and the low-rank one should be img.mat under oscillate
# 1. change data inputs from .h5 k-space data to .mat image space data
# 2. change each subject loading path by reading from three .txt files (Need to be Updated once get new data)
# 3. skip the ID and offer warning if corresponding img files cannot be found
# 4. squeezed each subject img from 120x120x4x16x1x1x1x24 to 120x120x4x16x24 and then zero padding to 128x128
# 5. divide each slice to two channels: real+imag; 
#    divide each (fullysampled - oscillate) slice to two channels as labels: real+imag

import os
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn.functional as F

class ArtifactImageSliceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, id_list_file: str, mode: str = 'train'):
        super().__init__()

        self.lowrank_paths = []
        self.clean_paths = []

        with open(id_list_file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]

        for sample_id in ids:
            lowrank_mat = Path(root_dir) / f"{sample_id}" / f"{sample_id}_B1000" / "mbmre_both" / "img.mat"
            clean_mat = Path(root_dir) / f"{sample_id}" / f"{sample_id}_B1000" / "mbmre" / "img.mat"
            if lowrank_mat.exists() and clean_mat.exists():
                self.lowrank_paths.append(str(lowrank_mat))
                self.clean_paths.append(str(clean_mat))
            else:
                print(f"Warning: Missing img.mat in {sample_id}, skipping.")

        self.samples = []
        for i in range(len(self.lowrank_paths)):
            lowrank = loadmat(self.lowrank_paths[i])['img']

            lowrank = np.squeeze(lowrank)  # [120, 120, 4, 16, 24]

            H, W, B, S, T = lowrank.shape #ZS save I/O time
            for t in range(T):
                for b in range(B):
                    for s in range(S):
                        self.samples.append((i, b, s, t))

        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, b, s, t = self.samples[idx]

        lowrank = loadmat(self.lowrank_paths[i])['img']
        clean = loadmat(self.clean_paths[i])['img']

        lowrank = np.squeeze(lowrank)  # [120, 120, 4, 16, 24]
        clean = np.squeeze(clean)

        slice_lowrank = lowrank[:, :, b, s, t].astype(np.complex64)
        slice_clean = clean[:, :, b, s, t].astype(np.complex64)
        slice_artifact = slice_lowrank - slice_clean #ZS

        scale = 1000000
        slice_lowrank = slice_lowrank * scale
        slice_artifact = slice_artifact * scale

        slice_lowrank = np.stack([np.real(slice_lowrank), np.imag(slice_lowrank)], axis=0)
        slice_artifact = np.stack([np.real(slice_artifact), np.imag(slice_artifact)], axis=0)
        # ZS Convert to tensor before padding
        slice_lowrank = torch.from_numpy(slice_lowrank).float()
        slice_artifact = torch.from_numpy(slice_artifact).float()
        # ZS Pad to 128x128 
        slice_lowrank = F.pad(slice_lowrank, (4,4,4,4), mode="constant", value=0)
        slice_artifact = F.pad(slice_artifact, (4,4,4,4), mode="constant", value=0)


        #print(slice_lowrank.shape); # shape test
        #print(slice_lowrank.dtype); # type test, should in x64
        #print(slice_clean.shape); # shape test
        #print(slice_clean.dtype); # type test, should in x64

        #slice_artifact = slice_lowrank - slice_clean #ZS

        #print(slice_artifact.shape); # shape test
        #print(slice_artifact.dtype); # type test

        #input_tensor = np.stack([np.real(slice_lowrank), np.imag(slice_lowrank)], axis=0)  # [2, 120, 120]
        #label_tensor = np.stack([np.real(slice_artifact), np.imag(slice_artifact)], axis=0)  # [2, 120, 120]

        return {
            'ispace_under': slice_lowrank,
            'ispace': slice_artifact
        }
