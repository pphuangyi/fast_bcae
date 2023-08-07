"""
"""

import copy
from math import log, exp
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

from rtid.datasets.dataset import DatasetTPC
from rtid.utils.utils import (get_norm_layer,
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


class Conv3dBlock(nn.Conv3d):

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

    def forward(self, data):
        return self.activ(self.norm(super().forward(self.pad(data))))


def logit(prob):
    return log(prob / (1. - prob))


class ProbPredictor(nn.Module):
    def __init__(self,
                 norm = 'batch',
                 activ={'name': 'leakyrelu',
                        'negative_slope': .1}):

        self.model = nn.ModuleList([
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

    def forward(self, data):
        """
        """
        mask = (data != 0)

        prob = data
        for layer in self.model:
            prob = layer(prob, mask)

        return prob


class Decoder(nn.Module):
    def __init__(self,
                 norm = 'batch',
                 activ={'name': 'leakyrelu',
                        'negative_slope': .1}):

        self.model = nn.Sequential(
            Conv3dBlock(1, 16,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(16, 16,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(16, 16,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(16, 8,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(8, 8,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(8, 8,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(8, 4,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(4, 2,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            nn.Conv3d(2, 1, kernel_size = 1)
        )

    def forward(self, data):
        return self.model(data)


class Model(nn.Module):
    def __init__(self,
                 norm = 'batch',
                 activ={'name': 'leakyrelu',
                        'negative_slope': .1},
                 # probability
                 prob = None,
                 prob_threshold = None,
                 alpha = 4.,
                 eps = 1e-6,
                 prob_min = 0,
                 prob_max = 1):

        super().__init__()

        if prob is None:

            self.logit_threshold = logit(prob_threshold)
            self.alpha = alpha
            self.eps = eps

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
        else:
            self.prob = prob

        self.decoder = nn.Sequential(
            Conv3dBlock(1, 16,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(16, 16,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(16, 16,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(16, 8,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(8, 8,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(8, 8,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(8, 4,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            Conv3dBlock(4, 2,
                        kernel_size = 3,
                        padding = 1,
                        norm = norm,
                        activ = activ),
            nn.Conv3d(2, 1, kernel_size = 1)
        )
        # self.prob_min = prob_min
        # self.prob_max = prob_max

    def forward(self, data, inference=False):

        mask = (data != 0)
        if hasattr(self, 'prob_predictor'):
            prob = data

            for layer in self.prob_predictor:
                prob = layer(prob, mask)

            logit = torch.logit(prob, eps = self.eps)
            if inference:
                gate = (logit - self.logit_threshold) > 0.
            else:
                logit_rnd = torch.logit(torch.rand_like(prob), eps = self.eps)
                gate_hard = logit - logit_rnd
                gate = torch.sigmoid(self.alpha * gate_hard)
                # gate = torch.sigmoid(self.alpha * (logit - logit_rnd))

            # prob = torch.clamp(prob,
            #                    min = self.prob_min,
            #                    max = self.prob_max) * mask
        else:
            prob = (self.prob * torch.ones_like(data)) * mask
            gate = (prob - torch.rand_like(prob)) > 0.

        # mask = (prob + torch.rand_like(prob)) > 1.
        # logit_rnd = torch.logit(torch.rand_like(prob), eps = self.eps)
        # gate = torch.sigmoid(self.alpha * (logit - logit_rnd))

        with torch.no_grad():
            reco_hard = self.decoder(data * (gate_hard > 0))

        reco = self.decoder(data * gate)

        return reco, prob, gate, reco_hard


def get_lr(optim):
    """
    Get the current learning rate
    """
    for param_group in optim.param_groups:
        return param_group['lr']


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

    loss_sum, mse_sum, mse_hard_sum = 0, 0, 0
    keep_sum, sig_sum = 0, 0

    for idx, adc in enumerate(dataloader):

        if idx >= batches_per_epoch:
            break

        # pad the z dimension to have length 256

        adc = adc.to(device)
        reco, prob, gate, reco_hard = model(adc, inference = (optimizer is None))

        mse = loss_fn(reco, adc)

        prob = prob.sum()
        true = (adc > 0).sum()

        # if (idx + 1) % 100 == 0:
        #     loss = prob / true
        # else:
        #     loss = mse
        # loss = mse
        prob_loss = prob / true
        prob_coef = 1000 * max(0, prob_loss.item() - .1)
        loss = mse + prob_coef * prob_loss

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.item()
        mse_sum += mse.item()

        mse_hard = loss_fn(reco_hard, adc)
        mse_hard_sum += mse_hard.item()

        keep_sum += gate.sum().item()
        sig_sum += true.item()

        pbar.update()
        postfix = {'loss': loss_sum / (idx + 1),
                   'mse': mse_sum / (idx + 1),
                   'mse hard': mse_hard_sum / (idx + 1),
                   'keep ratio': keep_sum / sig_sum}
        pbar.set_postfix(postfix)

    return postfix


MANIFEST_TRAIN = '/data/sphenix/auau/highest_framedata_3d/outer/train.txt'
MANIFEST_VALID = '/data/sphenix/auau/highest_framedata_3d/outer/test.txt'


def main():

    device = 'cuda'
    learning_rate = 1e-3
    num_warmup_epochs = 20
    num_epochs = 100
    sched_steps = 20
    sched_gamma = .95
    batch_size = 4
    batches_per_epoch = 500
    prob = None
    prob_threshold = .9
    # prob_min = .1
    # prob_max = .5

    # model and loss function
    model = Model(prob = prob,
                  prob_threshold = prob_threshold).to(device)
                  # prob_min = prob_min,
                  # prob_max = prob_max
    loss_fn = nn.MSELoss()

    # optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # schedular
    milestones = range(num_warmup_epochs, num_epochs, sched_steps)
    scheduler = MultiStepLR(optimizer,
                            milestones = milestones,
                            gamma = sched_gamma)

    # data loader

    dataset_train = DatasetTPC(MANIFEST_TRAIN, dimension = 3)
    dataset_valid = DatasetTPC(MANIFEST_VALID, dimension = 3)
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
        # with torch.no_grad():
        #     desc = f'Validation Epoch {epoch} / {num_epochs}'
        #     valid_stat = run_epoch(model,
        #                            loss_fn,
        #                            dataloader_valid,
        #                            desc = desc,
        #                            batches_per_epoch = batches_per_epoch,
        #                            device = device)
        # update learning rate
        scheduler.step()
        current_lr = get_lr(optimizer)
        print(f'current learning rate = {current_lr:.10f}')


main()
