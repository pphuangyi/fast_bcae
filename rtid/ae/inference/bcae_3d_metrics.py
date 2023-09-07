from pathlib import Path
from collections import defaultdict
import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

# from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders
from rtid.datasets.dataset import DatasetTPC
from neuralcompress.utils.load_bcae_models import (
    load_bcae_encoder,
    load_bcae_decoder
)

from neuralcompress.models.bcae_combine import BCAECombine


#################################################################
# =================== Compress and decompress ===================
# Load data
# data_path   = Path('/data/yhuang2/sphenix/auau/highest_framedata_3d/outer/')
# data_config = {
#     'batch_size' : 32,
#     'train_sz'   : 0,
#     'valid_sz'   : 0,
#     'test_sz'    : 320, # there are only 8 data files contained
#     'is_random'  : False,
#     'shuffle'    : False,
# }
# _, _, loader = get_tpc_dataloaders(data_path, **data_config)
DATA_ROOT = Path('/data/yhuang2/sphenix/auau/highest_framedata_3d/outer/')
dataset = DatasetTPC(DATA_ROOT, split = 'test', dimension = 3)
loader = DataLoader(dataset, batch_size = 32)

device = 'cuda'

# Load encoder
checkpoint_path = Path('/home/yhuang2/PROJs/NeuralCompression/checkpoints')
epoch           = 2000
encoder = load_bcae_encoder(checkpoint_path, epoch)
decoder = load_bcae_decoder(checkpoint_path, epoch)
encoder.to(device)
decoder.to(device)

# run compression and decompression
threshold = .52
combine = BCAECombine(threshold = threshold)
progbar = tqdm.tqdm(
    desc="BCAE recall study",
    total=len(loader),
    dynamic_ncols=True
)

T, P, TP = 0, 0, 0
mses = []
mse_mag_ratios = []

with torch.no_grad():
    for i, batch in enumerate(loader):
        batch  = batch.to(device)

        comp   = encoder(batch)
        comp   = comp.half() # we save the compressed result as half float
        decomp = combine(decoder(comp.float()))

        true = (batch > 0).sum().item()
        pos = (decomp > 0).sum().item()
        true_pos = ((decomp > 0) * (batch > 0)).sum().item()

        T += true
        P += pos
        TP += true_pos

        mse = torch.pow(decomp - batch, 2).mean().item()
        mag = torch.pow(batch, 2).mean().item()
        mses.append(mse)
        mse_mag_ratios.append(mse / mag)

        progbar.update()
    progbar.close()

recall = TP / T
precision = TP / P

print(f'recall: {recall:.3f}')
print(f'precision: {precision:.3f}')

mean = np.mean(mses)
std  = np.std(mse)
print(f'MSE: mean = {mean:.3f}, std = {std:.3f}')

mean = np.mean(mse_mag_ratios)
std  = np.std(mse_mag_ratios)
print(f'MSE_MAG_ratio: mean = {mean:.3f}, std = {std:.3f}')
