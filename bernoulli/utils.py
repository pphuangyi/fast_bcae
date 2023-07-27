"""
"""
import copy
import torch
from torch import nn

def extract_name_kwargs(obj):
    """
    """
    if isinstance(obj, dict):
        obj    = copy.copy(obj)
        name   = obj.pop('name')
        kwargs = obj
    else:
        name   = obj
        kwargs = {}

    return (name, kwargs)


def get_norm_layer(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch':
        return nn.BatchNorm3d(features, **kwargs)

    if name == 'instance':
        return nn.InstanceNorm3d(features, **kwargs)

    raise ValueError("Unknown Layer: '%s'" % name)


def get_activ_layer(activ):
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(**kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(**kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'sigmoid':
        return nn.Sigmoid()

    raise ValueError("Unknown activation: '%s'" % name)

