# preprocess_artifactnet.py
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from tqdm import tqdm
import gc

def save_processed_dataset(split_txt_path, root_dir, save_dir):
    with open(split_txt_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]

    # ZS 07/08 add circular mask
    #mask = loadmat("src/circular_mask_0_75.mat")['mask'] 
    #mask = np.pad(mask, ((4,4),(4,4)), mode='constant') 
    #mask = np.stack([mask]*2, axis=0) 
    #mask = torch.from_numpy(mask).float()

    #ids = ids[39:55]
    sample_idx = 0
    for sample_id in tqdm(ids):
        #print(f"\n>>> processing {sample_id}", flush=True)
        try:
            lowrank_path = Path(root_dir) / f"{sample_id}" / f"{sample_id}_B1000" / "mbmre_both" / "img_US.mat"
            clean_path   = Path(root_dir) / f"{sample_id}" / f"{sample_id}_B1000" / "mbmre" / "img.mat"
            if not (lowrank_path.exists() and clean_path.exists()):
                print(f"Skipping {sample_id} (missing file)")
                continue
            
            lowrank = loadmat(lowrank_path)['img']
            clean   = loadmat(clean_path)['img']
            lowrank = np.squeeze(lowrank) # [120, 120, 4, 16, 24]
            clean   = np.squeeze(clean)

            H, W, B, S, T = lowrank.shape
            for t in range(T):
                for b in range(B):
                    for s in range(S):
                        slc_lr = lowrank[:, :, b, s, t].astype(np.complex64)
                        slc_cl = clean[:, :, b, s, t].astype(np.complex64)
                        slc_artifact = slc_lr - slc_cl
                        
                        scale = 1000000
                        x = slc_lr * scale
                        y = slc_artifact * scale

                        x = torch.from_numpy(np.stack([np.real(x), np.imag(x)], axis=0)).float()
                        y = torch.from_numpy(np.stack([np.real(y), np.imag(y)], axis=0)).float()

                        x = F.pad(x, (4,4,4,4), mode="constant", value=0)  # pad to [2,128,128]
                        y = F.pad(y, (4,4,4,4), mode="constant", value=0)



                        # save
                        save_path = save_dir / f'sample_{sample_idx:06d}.pt'
                        with open(save_path, 'wb') as f:
                            #torch.save({'x': x, 'y': y, 'mask': mask}, f) #ZS: 07/08 add circular mask
                            torch.save({'x': x, 'y': y}, f)

                        sample_idx += 1

                        # cleanup
                        del x, y, slc_lr, slc_cl, slc_artifact
                        gc.collect()
        except Exception as e:
            print(f"Error when processing {sample_id}: {e}")
            continue

        del lowrank, clean
        gc.collect()


if __name__ == "__main__":
    root_dir = "/mnt/external/zhuoyu/fully+osci"
    data_dir = Path(__file__).parent / "data/v0"

    for split in ['training', 'validation', 'test']:
        print(f"\n>>> processing {split}")
        save_dir = Path(f'data_processed_artifact_removal/{split}')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_processed_dataset(data_dir / f"{split}.txt", root_dir, save_dir)
