"""
"""
import argparse
from itertools import chain
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import yaml

import torch
from torch import nn
from torch.nn.functional import pad
from torch.jit import trace, save
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

from rtid.datasets.dataset import DatasetTPC
from networks import Encoder, Decoder
from rtid.utils.runtime import runtime


MANIFEST_TRAIN = '/data/sphenix/auau/highest_framedata_3d/outer/train.txt'
MANIFEST_VALID = '/data/sphenix/auau/highest_framedata_3d/outer/test.txt'

def get_args(description):
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description)

    parser.add_argument('--transform',
                        action = 'store_true',
                        help   = ('if used, does the transform'))
    parser.add_argument('--num-encoder-layers',
                        dest    = 'num_encoder_layers',
                        type    = int,
                        default = 3,
                        help    = 'number of encoder layers (default = 3)')
    parser.add_argument('--num-decoder-layers',
                        dest    = 'num_decoder_layers',
                        type    = int,
                        default = 3,
                        help    = 'number of decoder layers (default = 3)')
    parser.add_argument('--num-epochs',
                        dest = 'num_epochs',
                        type = int,
                        help = 'number of epochs')
    parser.add_argument('--num-warmup-epochs',
                        dest = 'num_warmup_epochs',
                        type = int,
                        help = ('number of warmup epochs, '
                                'must be smaller than number of epochs'))
    parser.add_argument('--batches-per-epoch',
                        dest    = 'batches_per_epoch',
                        type    = int,
                        default = float('inf'),
                        help    = ('maximum number of batches per epoch, '
                                   '(default = inf)'))
    parser.add_argument('--sched-steps',
                        dest    = 'sched_steps',
                        type    = int,
                        default = 20,
                        help    = ('The steps for every decrease of '
                                   'learning rate. We will be using '
                                   'MultiStepLR scheduler, and we will '
                                   'multiply the learning rate by a gamma < 1 '
                                   'every [sched-steps] after reaching '
                                   '[num-warmup-epochs]. (default = 20)'))
    parser.add_argument('--sched-gamma',
                        dest    = 'sched_gamma',
                        type    = float,
                        default = .9,
                        help    = ('The gamma multiplied to learning rate. '
                                   'See help for [sched-steps] for more '
                                   'information. (default = .9)'))
    parser.add_argument('--device',
                        type    = str,
                        default = 'cuda',
                        choices = ('cuda', 'cpu'),
                        help    = ('device (default = cuda)'))
    parser.add_argument('--batch-size',
                        dest    = 'batch_size',
                        type    = int,
                        default = 4,
                        help    = 'batch size, (default = 4)')
    parser.add_argument('--learning-rate',
                        dest    = 'learning_rate',
                        type    = float,
                        default = 1e-3,
                        help    = 'learning rate, (default = 1e-3)')
    parser.add_argument('--save-frequency',
                        dest    = 'save_frequency',
                        type    = int,
                        default = 50,
                        help    = ('frequency of saving checkpoints, '
                                   '(default = 50)'))
    parser.add_argument('--checkpoint-path',
                        dest    = 'checkpoint_path',
                        type    = str,
                        default = './checkpoints',
                        help    = ('directory to save checkpoints, '
                                   '(default = ./checkpoints)'))

    return parser.parse_args()


def get_jit_input(tensor, batch_size, device):
    """
    Get a dummy input for jit tracing
    """
    dummy = torch.ones_like(tensor)
    shape = (batch_size, ) + (1, ) * tensor.dim()
    dummy = dummy.repeat(shape)
    return dummy.to(device)


def get_lr(optim):
    """
    Get the current learning rate
    """
    for param_group in optim.param_groups:
        return param_group['lr']


class CheckpointSaver:
    def __init__(self,
                 checkpoint_path,
                 frequency = -1,
                 benchmark = float('inf'),
                 prefix = 'mod'):

        self.checkpoint_path = Path(checkpoint_path)
        self.frequency = frequency
        self.benchmark = benchmark
        self.prefix = prefix

    def __call__(self, model, data, epoch, metric):

        traced_model = trace(model, data)

        name = self.checkpoint_path/f'{self.prefix}_last.pth'
        save(traced_model, name)

        if self.frequency > 0:
            if epoch % self.frequency == 0:
                name = self.checkpoint_path/f'{self.prefix}_{epoch}.pth'
                save(traced_model, name)

        if metric < self.benchmark:
            self.benchmark = metric
            name = self.checkpoint_path/f'{self.prefix}_best.pth'
            save(traced_model, name)


def run_epoch(encoder,
              decoder,
              loss_fn,
              dataloader,
              desc,
              optimizer=None,
              batches_per_epoch=float('inf'),
              device = 'cuda',
              transform = False):
    """
    """
    total = min(batches_per_epoch, len(dataloader))
    pbar = tqdm(desc = desc, total = total)
    total_loss = 0
    true, pos, true_pos = 0, 0, 0

    for idx, adc in enumerate(dataloader):

        if idx >= batches_per_epoch:
            break

        adc = adc.to(device)
        tag = adc > 0

        code = encoder(pad(adc, (0, 7)))
        output = decoder(code)
        output = output[..., :-7]

        if transform:
            output = torch.clamp(output, max=5.08)
            output = torch.exp(output) * 6 + 64

        loss = loss_fn(output, adc)
        total_loss += loss.item()

        with torch.no_grad():
            true += tag.sum().item()
            pos += (output > 0).sum().item()
            true_pos = (tag * (output > 0)).sum().item()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.update()
        result = {'loss': total_loss / (idx + 1),
                  'precision': true_pos / pos,
                  'recall': true_pos / true,}
        pbar.set_postfix(result)

    return result


def main():

    args = get_args('2d TPC Data Compression')

    transform          = args.transform
    num_encoder_layers = args.num_encoder_layers
    num_decoder_layers = args.num_decoder_layers
    num_epochs         = args.num_epochs
    num_warmup_epochs  = args.num_warmup_epochs
    batch_size         = args.batch_size
    batches_per_epoch  = args.batches_per_epoch
    sched_steps        = args.sched_steps
    sched_gamma        = args.sched_gamma
    device             = args.device
    learning_rate      = args.learning_rate
    save_frequency     = args.save_frequency
    checkpoints        = Path(args.checkpoint_path)

    # set up checkpoint folder and save config
    assert not checkpoints.exists()
    checkpoints.mkdir(parents = True)
    with open(checkpoints/'config.yaml', 'w') as config_file:
        yaml.dump(vars(args),
                  config_file,
                  default_flow_style = False)

    # model and loss function
    encoder = Encoder(16, num_encoder_layers, 3).to(device)
    decoder = Decoder(16, num_decoder_layers, 3).to(device)
    loss_fn = nn.MSELoss()

    # optimizer
    params = chain(encoder.parameters(), decoder.parameters())
    optimizer = AdamW(params, lr=learning_rate)

    # schedular
    milestones = range(num_warmup_epochs, num_epochs, sched_steps)
    scheduler = MultiStepLR(optimizer,
                            milestones = milestones,
                            gamma = sched_gamma)

    # data loader
    dataset_train = DatasetTPC(MANIFEST_TRAIN,
                               dimension = 2,
                               axis_order = ('layer', 'azimuth', 'beam'))
    dataset_valid = DatasetTPC(MANIFEST_VALID,
                               dimension = 2,
                               axis_order = ('layer', 'azimuth', 'beam'))
    dataloader_train = DataLoader(dataset_train,
                                  batch_size = batch_size,
                                  shuffle = True)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size = batch_size)

    # get dummy data for scripting
    data = dataset_train[0]
    dummy_input = get_jit_input(data, batch_size, device)
    with torch.no_grad():
        dummy_compr = encoder(dummy_input)

    # get inference time
    samples_per_second = runtime(encoder,
                                 input_shape = data.shape,
                                 batch_size = batch_size,
                                 num_inference_batches = 1000,
                                 script = True,
                                 device = device)
    print(f'samples per second = {samples_per_second: .1f}')

    ckpt_saver_enc = CheckpointSaver(checkpoints, save_frequency, prefix='enc')
    ckpt_saver_dec = CheckpointSaver(checkpoints, save_frequency, prefix='dec')

    df_data_train = defaultdict(list)
    df_data_valid = defaultdict(list)
    for epoch in range(1, num_epochs + 1):

        current_lr = get_lr(optimizer)
        print(f'current learning rate = {current_lr:.10f}')

        # train
        desc = f'Train Epoch {epoch} / {num_epochs}'
        train_stat = run_epoch(encoder,
                               decoder,
                               loss_fn,
                               dataloader_train,
                               desc              = desc,
                               optimizer         = optimizer,
                               batches_per_epoch = batches_per_epoch,
                               device            = device,
                               transform         = transform)

        # validation
        with torch.no_grad():
            desc = f'Validation Epoch {epoch} / {num_epochs}'
            valid_stat = run_epoch(encoder,
                                   decoder,
                                   loss_fn,
                                   dataloader_valid,
                                   desc              = desc,
                                   batches_per_epoch = batches_per_epoch,
                                   device            = device,
                                   transform         = transform)

        # save checkpoints
        metric = valid_stat['loss']
        ckpt_saver_enc(encoder, dummy_input, epoch, metric)
        ckpt_saver_dec(decoder, dummy_compr, epoch, metric)

        # save record
        for key, val in train_stat.items():
            df_data_train[key].append(val)
        df_data_train['lr'].append(current_lr)
        df_data_train['epoch'].append(epoch)

        for key, val in valid_stat.items():
            df_data_valid[key].append(val)
        df_data_valid['lr'].append(current_lr)
        df_data_valid['epoch'].append(epoch)

        df_train = pd.DataFrame(data = df_data_train)
        df_valid = pd.DataFrame(data = df_data_valid)
        df_train.to_csv(checkpoints/'train_log.csv', index = False, float_format='%.6f')
        df_valid.to_csv(checkpoints/'valid_log.csv', index = False, float_format='%.6f')

        # update learning rate
        scheduler.step()

main()
