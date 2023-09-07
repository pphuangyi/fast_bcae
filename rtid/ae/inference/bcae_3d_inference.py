"""
Inference time study
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path

import torch
from torch.jit import trace

from torchperf import perf

from rtid.datasets.dataset import DatasetTPC
from rtid.utils.runtime import runtime
from rtid.utils.utils import count_parameters

from neuralcompress.utils.load_bcae_models import load_bcae_encoder

torch.backends.cudnn.benchmark = True


def get_args(description):
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description,
                                     formatter_class = RawTextHelpFormatter)

    parser.add_argument('--batch-size',
                        dest    = 'batch_size',
                        type    = int,
                        default = 4,
                        help    = 'batch size, (default = 4)')
    parser.add_argument('--script',
                        action = 'store_true',
                        help   = 'whether to script the model')
    parser.add_argument('--half-mode',
                        dest    = 'half_mode',
                        default = None,
                        choices = ('none', 'tensor', 'autocast'),
                        help    = ('how to run in half precision.\n'
                                   "- input 'none' or omitting the flag: "
                                   "use full precision\n"
                                   "- 'tensor': halve the model weight and "
                                   "the input data direcly\n"
                                   "- 'autocast': use "
                                   "'torch.cuda.amp.autocast'"))
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

    args = get_args('3d TPC Data Compression inference time study')

    batch_size         = args.batch_size
    script             = args.script
    half_mode          = args.half_mode
    num_batches        = args.num_batches
    num_warmup_batches = args.num_warmup_batches
    device             = args.device
    gpu_id             = args.gpu_id
    if device == 'cuda':
        device = f'{device}:{gpu_id}'

    # model and data
    checkpoint_path = Path('/home/yhuang2/PROJs/NeuralCompression/checkpoints')
    encoder = load_bcae_encoder(checkpoint_path, 2000).to(device)
    encoder.eval()

    num_parameters = count_parameters(encoder)
    # print(f'number of parameters: {num_parameters / (1024 ** 2):.3f}M')


    input_shape = torch.Size([1, 192, 249, 16])
    data = torch.randn((batch_size, ) + input_shape).to(device)
    results = []
    @perf(o = results, n = num_batches, w = num_warmup_batches)
    def run(encoder, data):
        encoder(data)


    with torch.no_grad():
        if half_mode is None or half_mode == 'none':

            # print('Use full precision')
            if script:
                encoder = trace(encoder, data)

            run(encoder, data)

        elif half_mode == 'autocast':

            assert 'cuda' in device
            # print('Use half precision with autocast')
            if script:
                encoder = trace(encoder, data)
            with torch.cuda.amp.autocast():
                run(encoder, data)

        elif half_mode == 'tensor':

            # print('Use half precision by halving the model and the input')
            encoder.half()
            data_half = data.half()
            if script:
                encoder_half = trace(encoder, data_half)
            run(encoder_half, data_half)


    total_time = sum(results) / 1000.
    total_frames = num_batches * batch_size
    samples_per_second = total_frames / total_time
    print(f'{samples_per_second:.1f}')

main()
