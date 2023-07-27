"""
Load TPC data as multi-channel 2d dataset
"""

import numpy as np

import torch
from torch.utils.data import Dataset


class DatasetTPC2d(Dataset):
    """
    Load TPC data as multi-channel 2d dataset
    """
    def __init__(self, manifest):
        """
        Input
        ========
        manifest (str): The filename of the data manifest.
            Each line in file is an absolute path of a data file.
        """
        super().__init__()

        with open(manifest, 'r') as file_handle:
            self.fnames = file_handle.read().splitlines()

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        fname = self.fnames[index]
        datum = np.load(fname).astype(np.float32)
        # (azimuth, z, layer) -> (layer, azimuth, z)
        # adc = torch.tensor(datum).permute((2, 0, 1))

        return torch.tensor(datum).unsqueeze(0)


# from pathlib import Path
# ROOT = Path('/data/datasets/sphenix/highest_framedata_3d/outer')
# def test():
#
#     for split in ('train', 'test'):
#         manifest = ROOT/f'{split}.txt'
#         dataset = DatasetTPC2d(manifest)
#         print(f'The dataset contains {len(dataset)} samples.')
#         for adc in dataset:
#             print('adc', adc.shape, adc.type())
#             break
#
# test()
