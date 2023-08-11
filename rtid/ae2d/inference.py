"""
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
import argparse

import torch
from torch.jit import trace

from torchperf import perf

from rtid.datasets.dataset import DatasetTPC
from rtid.utils.runtime import runtime
from rtid.utils.utils import count_parameters

from rtid.ae2d.bcae.networks import Encoder

torch.backends.cudnn.benchmark = True

DATA_ROOT = '/data/yhuang2/sphenix/auau/highest_framedata_3d/outer/'


def get_args(description):
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description)

    parser.add_argument('--num-encoder-layers',
                        dest    = 'num_encoder_layers',
                        type    = int,
                        default = 3,
                        help    = 'number of encoder layers (default = 3)')
    parser.add_argument('--batch-size',
                        dest    = 'batch_size',
                        type    = int,
                        default = 4,
                        help    = 'batch size, (default = 4)')
    parser.add_argument('--script',
                        action = 'store_true',
                        help   = 'whether to script the model')
    parser.add_argument('--half',
                        action = 'store_true',
                        help   = 'whether to use half precision')
    parser.add_argument('--num-batches',
                        dest    = 'num_batches',
                        type    = int,
                        default = 1000,
                        help    = 'number of batches, (default = 1000)')
    parser.add_argument('--num-warmup-batches',
                        dest    = 'num_warmup_batches',
                        type    = int,
                        default = 10,
                        help    = 'number of warmup batches, (default = 10)')
    parser.add_argument('--device',
                        type    = str,
                        default = 'cuda',
                        choices = ('cuda', 'cpu'),
                        help    = ('device (default = cuda)'))
    parser.add_argument('--gpu-id',
                        dest    = 'gpu_id',
                        type    = int,
                        default = 0,
                        help    = ('ID of GPU card. Only effective when '
                                   'device is cuda (default = 0)'))

    return parser.parse_args()



def main():

    args = get_args('2d TPC Data Compression inference time study')

    num_encoder_layers = args.num_encoder_layers
    batch_size         = args.batch_size
    script             = args.script
    half               = args.half
    num_batches        = args.num_batches
    num_warmup_batches = args.num_warmup_batches
    device             = args.device
    gpu_id             = args.gpu_id
    if device == 'cuda':
        device = f'{device}:{gpu_id}'

    # model and data
    encoder = Encoder(16, num_encoder_layers, 3).to(device)
    encoder.eval()
    num_parameters = count_parameters(encoder)
    print(f'number of parameters: {num_parameters / (1024 ** 2):.3f}M')


    input_shape = torch.Size([16, 192, 256])
    data = torch.randn((batch_size, ) + input_shape).to(device)
    if script:
        encoder = trace(encoder, data)

    results = []
    @perf(o = results, n = num_batches, w = num_warmup_batches)
    def run():
        encoder(data)

    with torch.no_grad():
        if half and 'cuda' in device:
            print('Use half precision')
            with torch.cuda.amp.autocast():
                # encoder = encoder.half()
                run()
        else:
            print('Use full precision')
            run()

    total_time = sum(results) / 1000.
    total_frames = num_batches * batch_size
    samples_per_second = total_frames / total_time
    print(f'samples per second = {samples_per_second: .1f}')

main()
