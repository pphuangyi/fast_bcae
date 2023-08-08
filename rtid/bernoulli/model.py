"""
Drop umimportant signals, and try to reconstruct
the frame with the remaining signals.
"""

from math import log
from functools import partial

import torch
from torch import nn

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


class ProbPredictor(nn.Module):
    def __init__(self, norm, activ):

        super().__init__()

        layer = partial(SparseConv3d, norm = norm, activ = activ)

        self.model = nn.ModuleList([
            layer(1, 2, kernel_size = 3, padding = 1),
            layer(2, 2, kernel_size = 3, padding = 2, dilation = 2),
            layer(2, 2, kernel_size = 3, padding = 1),
            layer(2, 2, kernel_size = 3, padding = 2, dilation = 2),
            SparseConv3d(2, 1, kernel_size = 1, norm = None, activ = 'sigmoid')
        ])

    def forward(self, data, mask = None):
        """
        """
        if mask is None:
            mask = (data != 0)

        prob = data
        for layer in self.model:
            prob = layer(prob, mask)

        return prob


class Decoder(nn.Module):
    def __init__(self, norm, activ):

        super().__init__()

        layer = partial(Conv3dBlock, norm = norm, activ = activ)

        self.model = nn.Sequential(
            layer( 1, 16, kernel_size = 3, padding = 1),
            layer(16, 16, kernel_size = 3, padding = 1),
            layer(16, 16, kernel_size = 3, padding = 1),
            layer(16,  8, kernel_size = 3, padding = 1),
            layer( 8,  8, kernel_size = 3, padding = 1),
            layer( 8,  8, kernel_size = 3, padding = 1),
            layer( 8,  4, kernel_size = 3, padding = 1),
            layer( 4,  2, kernel_size = 3, padding = 1),
            nn.Conv3d(2, 1, kernel_size = 1)
        )

    def forward(self, data):
        return self.model(data)


class Model(nn.Module):
    """
    """
    def __init__(self,
                 norm = 'batch',
                 activ={'name': 'leakyrelu',
                        'negative_slope': .1},
                 prob = None,
                 alpha = 4.,
                 eps = 1e-6):

        super().__init__()

        if prob is None:
            self.alpha = alpha
            self.eps = eps
            self.prob_predictor = ProbPredictor(norm, activ)
        else:
            self.prob = prob

        self.decoder = Decoder(norm, activ)


    def forward(self, data, return_hard = False):
        """
        """

        mask = (data != 0)

        if hasattr(self, 'prob_predictor'):
            prob = self.prob_predictor(data, mask)
            logit = torch.logit(prob, eps = self.eps)
        else:
            logit = log(self.prob / (1. - self.prob))
            logit = logit * torch.ones_like(data) * mask

        logit_rnd = torch.logit(torch.rand_like(prob), eps = self.eps)
        logit_dff = logit - logit_rnd

        gate = torch.sigmoid(self.alpha * logit_dff)
        reco = self.decoder(data * gate)

        result = {'prob': prob, 'gate': gate, 'reco': reco}

        if return_hard:
            gate_hard = logit_dff > 0

            with torch.no_grad():
                reco_hard = self.decoder(data * gate_hard)

            result['gate_hard'] = gate_hard
            result['reco_hard'] = reco_hard

        return result
