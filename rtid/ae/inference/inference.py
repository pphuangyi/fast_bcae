"""
Inference time study
"""
import os
import argparse
from argparse import RawTextHelpFormatter

import torch
from torch.jit import trace

from torchperf import perf

from rtid.ae.model.network3d import Encoder as Encoder3d
from rtid.ae.model.network2d import Encoder as Encoder2d

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True


def get_args(description):
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description,
                                     formatter_class = RawTextHelpFormatter)

    parser.add_argument('--model-type',
                        dest    = 'model_type',
                        type    = str,
                        help    = ('model type, choose from 3d, 3d-fast, '
                                   'and 2d-[l] where l is the number of encoder layers'))
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
                        help    = 'device (default = cuda)')
    parser.add_argument('--gpu-id',
                        dest    = 'gpu_id',
                        type    = int,
                        default = 0,
                        help    = ('ID of GPU card. Only effective when '
                                   'device is cuda (default = 0)'))

    return parser.parse_args()


def main():

    args = get_args('BCAE inference speed study')

    model_type         = args.model_type
    batch_size         = args.batch_size
    script             = args.script
    half_mode          = args.half_mode
    num_batches        = args.num_batches
    num_warmup_batches = args.num_warmup_batches

    device = args.device
    if device == 'cuda':
        torch.cuda.set_device(args.gpu_id)

    # construct model and data
    if model_type == '3d':
        encoder = Encoder3d(conv_features = (8, 16, 32, 32)).to(device)
        input_shape = torch.Size([1, 16, 192, 256])
    elif model_type == '3d-fast':
        encoder = Encoder3d(conv_features = (2, 4, 4, 8)).to(device)
        input_shape = torch.Size([1, 16, 192, 256])
    elif model_type.startswith('2d'):
        num_encoder_layers = int(model_type.split('-')[-1])
        encoder = Encoder2d(16, num_encoder_layers, 3).to(device)
        input_shape = torch.Size([16, 192, 256])
    else:
        raise TypeError(f'unknow model type: {model_type}')

    encoder.eval()
    data = torch.randn((batch_size, ) + input_shape).to(device)

    # throughput calculation
    results = []
    @perf(o = results, n = num_batches, w = num_warmup_batches)
    def run(encoder, data):
        encoder(data)

    with torch.no_grad():
        if half_mode is None or half_mode == 'none':
            # Use full precision
            if script:
                encoder = trace(encoder, data)

            run(encoder, data)

        elif half_mode == 'autocast':

            assert 'cuda' in device
            # Use autocast
            if script:
                encoder = trace(encoder, data)
            with torch.cuda.amp.autocast():
                run(encoder, data)

        elif half_mode == 'tensor':
            # Use half precision
            encoder = encoder.half()
            data = data.half()
            if script:
                encoder = trace(encoder, data)
            run(encoder, data)

    total_time = sum(results) / 1000.
    total_frames = num_batches * batch_size
    samples_per_second = total_frames / total_time
    print(f'{samples_per_second:.1f}')

main()
