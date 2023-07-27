"""
"""
import copy
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import DatasetTPC2d
from utils import (get_norm_layer,
                   get_activ_layer)


class SparseConv3d(nn.Conv3d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding = 0,
                 norm = None,
                 activ = None,
                 **kwargs):

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         **kwargs)

        self.pad = nn.ReflectionPad3d(padding)
        self.norm = get_norm_layer(norm, out_channels)
        self.activ = get_activ_layer(activ)

    def forward(self, data, mask = None):
        if mask is None:
            mask = (data != 0)
        data = self.activ(self.norm(super().forward(self.pad(data))))
        return data * mask


class Model(nn.Module):
    def __init__(self,
                 norm = 'batch',
                 activ={'name': 'leakyrelu',
                        'negative_slope': .1}):

        super().__init__()

        self.prob_predictor = nn.ModuleList([
            SparseConv3d(1, 2,
                         kernel_size = 3,
                         padding = 1,
                         norm = norm,
                         activ = activ),
            SparseConv3d(2, 2,
                         kernel_size = 3,
                         padding = 2,
                         dilation = 2,
                         norm = norm,
                         activ = activ),
            SparseConv3d(2, 2,
                         kernel_size = 3,
                         padding = 1,
                         norm = norm,
                         activ = activ),
            SparseConv3d(2, 2,
                         kernel_size = 3,
                         padding = 2,
                         dilation = 2,
                         norm = norm,
                         activ = activ),
            SparseConv3d(2, 1,
                         kernel_size = 1,
                         # bias=False,
                         norm = None,
                         activ = 'sigmoid'),
        ])

        self.decoder = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(1, 2, kernel_size = 3),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(negative_slope = .1),
            nn.ReflectionPad3d(1),
            nn.Conv3d(2, 2, kernel_size = 3, dilation = 1),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(negative_slope = .1),
            nn.ReflectionPad3d(1),
            nn.Conv3d(2, 2, kernel_size = 3),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(negative_slope = .1),
            nn.ReflectionPad3d(1),
            nn.Conv3d(2, 2, kernel_size = 3, dilation = 1),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(negative_slope = .1),
            nn.Conv3d(2, 1, kernel_size = 1),
        )

    def forward(self, data):

        mask = (data != 0)
        prob = data

        for layer in self.prob_predictor:
             prob = layer(prob, mask)

        reco = self.decoder(data * self.get_mask(prob))

        return prob, reco

    @staticmethod
    def get_mask(prob):
        return (prob + torch.rand_like(prob)) > 1.


def run_epoch(model,
              loss_fn,
              dataloader,
              desc,
              optimizer=None,
              batches_per_epoch=float('inf'),
              device = 'cuda'):
    """
    """
    total = min(batches_per_epoch, len(dataloader))
    pbar = tqdm(desc = desc, total = total)

    loss_sum = 0
    mse_sum, exp_sum = 0, 0
    prob_sum = 0
    true_sum = 0

    for idx, adc in enumerate(dataloader):

        if idx >= batches_per_epoch:
            break

        # pad the z dimension to have length 256

        adc = adc.to(device)
        prob, reco = model(adc)

        mse = loss_fn(reco, adc)

        prob = prob.sum()
        true = (adc > 0).sum()

        # if (idx + 1) % 100 == 0:
        #     loss = prob / true
        # else:
        #     loss = mse
        loss = mse

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.item()
        mse_sum += mse.item()

        prob_sum += prob.item()
        true_sum += true.item()

        pbar.update()
        postfix = {'loss': loss_sum / (idx + 1),
                   'mse': mse_sum / (idx + 1),
                   'exp': prob_sum / true_sum}
        pbar.set_postfix(postfix)

    return postfix


MANIFEST_TRAIN = '/data/datasets/sphenix/highest_framedata_3d/outer/train.txt'
MANIFEST_VALID = '/data/datasets/sphenix/highest_framedata_3d/outer/test.txt'

def main():

    device = 'cuda'
    learning_rate = 1e-2
    num_epochs = 100
    batch_size = 4
    batches_per_epoch = 500

    # model and loss function
    model = Model().to(device)
    loss_fn = nn.MSELoss()

    # optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # # schedular
    # milestones = range(num_warmup_epochs, num_epochs, sched_steps)
    # scheduler = MultiStepLR(optimizer,
    #                         milestones = milestones,
    #                         gamma = sched_gamma)

    # data loader

    dataset_train = DatasetTPC2d(MANIFEST_TRAIN)
    dataset_valid = DatasetTPC2d(MANIFEST_VALID)
    dataloader_train = DataLoader(dataset_train,
                                  batch_size = batch_size,
                                  shuffle = True)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size = batch_size)


    for epoch in range(1, num_epochs + 1):

        # train
        desc = f'Train Epoch {epoch} / {num_epochs}'
        train_stat = run_epoch(model,
                               loss_fn,
                               dataloader_train,
                               desc = desc,
                               optimizer = optimizer,
                               batches_per_epoch = batches_per_epoch,
                               device = device)

        # validation
        with torch.no_grad():
            desc = f'Validation Epoch {epoch} / {num_epochs}'
            valid_stat = run_epoch(model,
                                   loss_fn,
                                   dataloader_valid,
                                   desc = desc,
                                   batches_per_epoch = batches_per_epoch,
                                   device = device)


main()
