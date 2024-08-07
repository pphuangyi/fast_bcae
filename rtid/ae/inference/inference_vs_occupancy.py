"""
Inference time study
"""
import os
import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import torch
from torch.nn.functional import pad

from torchperf import perf

from spoi.datasets.dataset_tpc import DatasetTPC
from rtid.ae.model.network3d import Encoder as Encoder3d
from rtid.ae.model.network2d import Encoder as Encoder2d

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True

DATA_ROOT = '/data/yhuang2/sphenix/auau/highest_framedata_3d/outer/'
RESULT_PATH = Path('throughput_vs_occupancy_results/')
if not RESULT_PATH.exists():
    RESULT_PATH.mkdir(parents=True)


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
                        default = 100,
                        help    = 'number of batches, (default = 100)')
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
    parser.add_argument('--gpu-name',
                        type    = str,
                        default = None,
                        help    = ('Name of the GPU card. '
                                   'If not given, use gpu_id. '
                                   '(default = None)'))

    return parser.parse_args()


def main():

    args = get_args('BCAE inference speed study')

    model_type         = args.model_type
    batch_size         = args.batch_size
    half_mode          = args.half_mode
    num_batches        = args.num_batches
    num_warmup_batches = args.num_warmup_batches

    device   = args.device
    gpu_id   = args.gpu_id
    gpu_name = args.gpu_name
    if gpu_name is None:
        gpu_name = str(gpu_id)

    if device == 'cuda':
        torch.cuda.set_device(gpu_id)


    # construct model and data
    axis_order = ('layer', 'azimuth', 'beam')

    if model_type == '3d':
        encoder = Encoder3d(conv_features = (8, 16, 32, 32)).to(device)
        dataset = DatasetTPC(DATA_ROOT,
                             split      = 'test',
                             dimension  = 3,
                             axis_order = axis_order)
        input_shape = torch.Size([1, 16, 192, 256])
    elif model_type == '3d-fast':
        encoder = Encoder3d(conv_features = (2, 4, 4, 8)).to(device)
        dataset = DatasetTPC(DATA_ROOT,
                             split      = 'test',
                             dimension  = 3,
                             axis_order = axis_order)
        input_shape = torch.Size([1, 16, 192, 256])
    elif model_type.startswith('2d'):
        num_encoder_layers = int(model_type.split('-')[-1])
        encoder = Encoder2d(16, num_encoder_layers, 3).to(device)
        dataset = DatasetTPC(DATA_ROOT,
                             split      = 'test',
                             dimension  = 2,
                             axis_order = axis_order)
        input_shape = torch.Size([16, 192, 256])
    else:
        raise TypeError(f'unknow model type: {model_type}')


    # if model_type.startswith('3d'):
    #     input_shape = torch.Size([1, 16, 192, 256])
    # else:
    #     input_shape = torch.Size([16, 192, 256])

    encoder = torch.compile(encoder)

    encoder.eval()

    stats = {'occupancy': [], 'throughput': []}

    # throughput calculation
    size = 16 * 249 * 192

    pbar = tqdm(dataset, desc='inference', total=len(dataset))
    for data in pbar:


        data = pad(data, (0, 7)).to(device)
        occupancy = (data > 0).sum().item() / size

        repeat_shape = (batch_size, ) + (1, ) * data.dim()
        batch = data.repeat(repeat_shape)

        # batch = torch.randn((batch_size, ) + input_shape).to(device)

        results = []
        @perf(o = results, n = num_batches, w = num_warmup_batches)
        def run(encoder, batch):
            encoder(batch)

        with torch.no_grad():
            if half_mode is None or half_mode == 'none':
                # Use full precision
                run(encoder, batch)

            elif half_mode == 'autocast':
                # Use autocast
                assert 'cuda' in device
                with torch.cuda.amp.autocast():
                    run(encoder, batch)

            elif half_mode == 'tensor':
                # Use half precision
                encoder = encoder.half()
                data = data.half()
                run(encoder, batch)

        total_time = sum(results) / 1000.
        total_frames = num_batches * batch_size
        throughput = total_frames / total_time

        stats['occupancy'].append(occupancy)
        stats['throughput'].append(throughput)

        postfix = {'occupancy': occupancy,
                   'throughput': throughput}

        pbar.set_postfix(postfix)
        pbar.update()

    df = pd.DataFrame(data=stats)
    fname = f'model_{model_type}-bs_{batch_size}-gpu_{gpu_name}_random.csv'
    df.to_csv(RESULT_PATH/fname, index = False, float_format='%.6f')


if __name__ == '__main__':
    main()
