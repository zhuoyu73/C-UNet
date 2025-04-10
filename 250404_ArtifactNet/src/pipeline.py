### artifactnet
# ZS
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.utils import make_grid
from tqdm import tqdm

from .models.artifactnet import ArtifactNet
from .datasets import ArtifactImageSliceDataset
from . import utils

# ZS made the changes below: 
# 1. changed the RACUNet_PAFT_RACUNet module (AFT block) to the ArtifactNet under models
#    a). removed the AFT block and reserves the CUNet block
#    b). changed input and output to complex number single slice image, as the CUNet only supports slicewise operations, 
#       so the input channel is now 2 and output channel is also 2 (complex number); the input to the model should be 
#       complex number image in size 120x120
# 2. changed the pipeline
#    a). substitute the model to the ArtifactNet model we defined in 1
#    b). changed the data input to isp_in, isp_true = batch['ispace_under'], batch['ispace']
#    c). set the label as label = isp_in - isp_true (undersampled img - fullysampled img)
#    d). changed the loss function to complex number MSE and metrics to MSE only
#    e). removed changing output img to magnitude and calculating RSS loss
#    f). data input
#           1) each slice now is distributed to [2, 120, 120], channel 0 is real and channel 1 is imag
#           2) intentionally set batch size = 64, as each img volume has 64 slices


class Pipeline:
    def __init__(self, save_dir: Path, debug: bool = False, verbose: bool = True) -> None:
        self.logger = utils.get_logger(verbose, save_dir)
        self.writer = SummaryWriter(save_dir)

        self.logger.info(f'seed: 42')
        utils.setup_seed(42)

        self.logger.info(f'PID = {os.getpid()}')

        self.train_loader, self.val_loader, self.test_loader = self._get_loader()

        self.model = ArtifactNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=1, eta_min=1e-5)

        self.debug = debug
        if self.debug:
            self.logger.info('debug mode')
        self.verbose = verbose
        self.save_dir = save_dir
        self.best_loss = float('inf') 

    
    @staticmethod
    def _get_loader():
        root_dir = '/mnt/external/zhuoyu/fully+osci'
        data_dir = Path(__file__).parent.parent / 'data/v0'
        train_dataset = ArtifactImageSliceDataset(root_dir, data_dir / 'training.txt')
        val_dataset   = ArtifactImageSliceDataset(root_dir, data_dir / 'validation.txt')
        test_dataset  = ArtifactImageSliceDataset(root_dir, data_dir / 'test.txt')
        print(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}, Test set: {len(test_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
        return train_loader, val_loader, test_loader

    def __call__(self):
        for self.epoch in range(50):
            self.logger.info(f'Epoch {self.epoch}')
            self.train_epoch()
            self.val_epoch()
            if self.debug and self.epoch == 1:
                break
        self.test_epoch()
        self.writer.close()

    def train_epoch(self):
        self.model.train()
        iter = len(self.train_loader)
        losses = {'mse_loss': []}

        for i, batch in enumerate(tqdm(self.train_loader, desc='Train', disable=not self.verbose, dynamic_ncols=True)):
            if self.debug and i == 2:
                break

            batch: dict[str, torch.Tensor]
            isp_in, isp_true = batch['ispace_under'], batch['ispace']

            pred = self.model(isp_in.to(self.device))
            label = (isp_in - isp_true).to(self.device)
            loss = F.mse_loss(pred, label)

            losses['mse_loss'].append(loss.item())
            self.writer.add_scalar('mse_loss/train', loss.item(), self.epoch * iter + i)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('lr/train', self.scheduler.get_last_lr()[0], self.epoch * iter + i)
            self.scheduler.step(self.epoch + i / iter)

        self.logger.info(f"Train: mse_loss = {np.mean(losses['mse_loss']):.4e}")
    
    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        losses = []

        for i, batch in enumerate(tqdm(self.val_loader, desc='Val', disable=not self.verbose, dynamic_ncols=True)):
            if self.debug and i == 2:
                break
            batch: dict[str, torch.Tensor]
            isp_in, isp_true = batch['ispace_under'], batch['ispace']
            pred = self.model(isp_in.to(self.device))
            label = (isp_in - isp_true).to(self.device)
            loss = F.mse_loss(pred, label)
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        self.logger.info(f"Val: mse_loss = {avg_loss:.4e}")
        self.writer.add_scalar('mse_loss/val', avg_loss, self.epoch)
        if not hasattr(self, 'best_loss') or avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save(self.model.state_dict(), self.save_dir / 'best_model.pt')
            self.logger.info(f"New best model saved with val_loss = {avg_loss:.4e}")

    
    @torch.no_grad()
    def test_epoch(self):
        self.model.load_state_dict(torch.load(self.save_dir / 'best_model.pt', map_location=self.device))
        self.model.eval()
        losses = []

        for i, batch in enumerate(tqdm(self.test_loader, desc='Test', disable=not self.verbose, dynamic_ncols=True)):
            if self.debug and i == 2:
                break
            batch: dict[str, torch.Tensor]
            isp_in, isp_true = batch['ispace_under'], batch['ispace']
            pred = self.model(isp_in.to(self.device))
            label = (isp_in - isp_true).to(self.device)
            loss = F.mse_loss(pred, label)
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        self.logger.info(f"Test: mse_loss = {avg_loss:.4e}")
        pd.DataFrame({'mse_loss': losses}).to_csv(self.save_dir / 'metrics.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    save_dir = Path('runs_artifactnet')
    save_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(save_dir=save_dir, debug=args.debug)
    pipeline()


if __name__ == '__main__':
    main()
