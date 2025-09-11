# compute_avg_max_signal.py

import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import argparse

def compute_average_max_signal(split_txt_path, root_dir):
    with open(split_txt_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]

    max_vals = []
    for sample_id in tqdm(ids, desc="Computing max(abs) per slice"):
        try:
            lowrank_path = Path(root_dir) / f"{sample_id}" / f"{sample_id}_B1000" / "mbmre_both" / "img_US.mat"
            clean_path   = Path(root_dir) / f"{sample_id}" / f"{sample_id}_B1000" / "mbmre" / "img.mat"

            if not (lowrank_path.exists() and clean_path.exists()):
                print(f"Skipping {sample_id} (missing file)")
                continue

            lowrank = np.squeeze(loadmat(lowrank_path)['img'])  # shape [H, W, B, S, T]

            H, W, B, S, T = lowrank.shape
            for b in range(B):
                for s in range(S):
                    for t in range(T):
                        slc = lowrank[:, :, b, s, t].astype(np.complex64)
                        max_val = np.max(np.abs(slc))
                        max_vals.append(max_val)

            del lowrank

        except Exception as e:
            print(f"Error when processing {sample_id}: {e}")
            continue

    avg_max = np.mean(max_vals)
    print(f"\n✅ Total slices processed: {len(max_vals)}")
    print(f"📊 Average max(abs) per slice: {avg_max:.3e}")
    return avg_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Root data folder, e.g., /mnt/external/zhuoyu/fully+osci')
    parser.add_argument('--split', type=str, required=True, choices=['training', 'validation', 'test'], help='Split to compute on')
    args = parser.parse_args()

    txt_path = Path(__file__).parent / "data/v0" / f"{args.split}.txt"
    avg_max = compute_average_max_signal(txt_path, args.root)
