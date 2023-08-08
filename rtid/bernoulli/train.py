"""
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

import argparse
from argparse import RawTextHelpFormatter
from collections import defaultdict
from pathlib import Path
import yaml
from tqdm import tqdm
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

from rtid.datasets.dataset import DatasetTPC
from rtid.utils.runtime import runtime
from rtid.utils.utils import get_lr, get_jit_input
from rtid.utils.checkpoint_saver import CheckpointSaver

from rtid.bernoulli.model import Model

DATA_ROOT = Path('/data/yhuang2/sphenix/auau/highest_framedata_3d/outer/')

def get_args(description):
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description,
                                     formatter_class = RawTextHelpFormatter)

    parser.add_argument('--prob',
                        type = float,
                        default = None,
                        help = ("If 'prob' is a float number between 0 and 1, "
                                "a signal (non-zero voxel) will be kept with "
                                "probability 'prob'. In this case, the model "
                                "only has the decoder part, but not the "
                                "probability predicting part. If 'prob' is "
                                "'None', the model will have two parts: "
                                "probability prediction and decoding. "
                                "(default = None)"))
    parser.add_argument('--prob-lambda',
                        dest = 'prob_lambda',
                        type = float,
                        default = 1000,
                        help = ('The coefficient to the probability loss '
                                '(default = 1000). For more detail, see '
                                "help information for 'prob-lower-bound'."))
    parser.add_argument('--prob-lower-bound',
                        dest = 'prob_lower_bound',
                        type = float,
                        default = .1,
                        help = ('The coefficient to the probability loss has '
                                'the following formula: \n\t\t prob-lambda * '
                                'max(0, prob. loss - prob-lower-bound). \n'
                                'The rational behind this formula is that, '
                                'when the probability of keeping a signal is '
                                "low enough, we don't need to lower it "
                                "any more. (default = .1)"))
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
    parser.add_argument('--num-epochs',
                        dest = 'num_epochs',
                        type = int,
                        help = 'number of epochs')
    parser.add_argument('--num-warmup-epochs',
                        dest = 'num_warmup_epochs',
                        type = int,
                        help = ('number of warmup epochs, '
                                'must be smaller than number of epochs'))
    parser.add_argument('--batch-size',
                        dest    = 'batch_size',
                        type    = int,
                        default = 4,
                        help    = 'batch size, (default = 4)')
    parser.add_argument('--batches-per-epoch',
                        dest    = 'batches_per_epoch',
                        type    = int,
                        default = float('inf'),
                        help    = ('maximum number of batches per epoch, '
                                   '(default = inf)'))
    parser.add_argument('--learning-rate',
                        dest    = 'learning_rate',
                        type    = float,
                        default = 1e-3,
                        help    = 'learning rate, (default = 1e-3)')
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
                        default = .95,
                        help    = ('The gamma multiplied to learning rate. '
                                   'See help for [sched-steps] for more '
                                   'information. (default = .95)'))
    parser.add_argument('--save-frequency',
                        dest    = 'save_frequency',
                        type    = int,
                        default = 10,
                        help    = ('frequency of saving checkpoints, '
                                   '(default = 10)'))
    parser.add_argument('--checkpoint-path',
                        dest    = 'checkpoint_path',
                        type    = str,
                        default = './checkpoints',
                        help    = ('directory to save checkpoints, '
                                   '(default = ./checkpoints)'))

    return parser.parse_args()


def run_epoch(model,
              loss_fn,
              dataloader, *,
              prob_lambda       = 1000,
              prob_lower_bound  = .1,
              desc              = '',
              optimizer         = None,
              batches_per_epoch = float('inf'),
              device            = 'cuda'):
    """
    """
    pbar = tqdm(desc = desc, total = min(batches_per_epoch, len(dataloader)))

    total = defaultdict(float)

    for idx, adc in enumerate(dataloader, 1):

        if idx >= batches_per_epoch:
            break

        adc = adc.to(device)
        result = model(adc, return_hard = True)

        # loss
        true = (adc > 0).sum()
        reco = result['reco']
        prob = result['prob']

        reco_loss = loss_fn(reco, adc)
        prob_loss = prob.sum() / true
        prob_coef = prob_lambda * max(0., prob_loss.item() - prob_lower_bound)
        loss = reco_loss + prob_coef * prob_loss

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # metrics
        total['reco_loss'] += reco_loss.item()
        total['prob_loss'] += prob_loss.item()
        total['loss'] += loss.item()

        gate = result['gate']
        gate_hard = result['gate_hard']
        reco_hard = result['reco_hard']

        total['gate'] += gate.sum().item()
        total['gate_hard'] += gate_hard.sum().item()
        total['reco_hard'] += loss_fn(reco_hard, adc).item()
        total['signal'] += true.item()

        # update the porgress bar
        pbar.update()
        postfix = {'loss': total['loss'] / idx,
                   'reco loss': total['reco_loss'] / idx,
                   'prob loss': total['prob_loss'] / idx,
                   'keep ratio': total['gate'] / total['signal'],
                   'reco error': total['reco_hard'] / idx,
                   'keep ratio hard': total['gate_hard'] / total['signal'],
                   'prob loss coef': prob_coef}
        pbar.set_postfix(postfix)

    return postfix


def main():

    args = get_args('Bernoulli Data Compression')

    # model specific parameters
    prob             = args.prob
    prob_lambda      = args.prob_lambda
    prob_lower_bound = args.prob_lower_bound

    # training device
    device = args.device
    gpu_id = args.gpu_id
    if device == 'cuda':
        device = f'{device}:{gpu_id}'

    # training and model saving parameters
    num_epochs        = args.num_epochs
    num_warmup_epochs = args.num_warmup_epochs
    batch_size        = args.batch_size
    batches_per_epoch = args.batches_per_epoch
    learning_rate     = args.learning_rate
    sched_steps       = args.sched_steps
    sched_gamma       = args.sched_gamma
    save_frequency    = args.save_frequency
    checkpoints       = Path(args.checkpoint_path)

    # set up checkpoint folder and save config
    # assert not checkpoints.exists()
    checkpoints.mkdir(parents = True, exist_ok = True)
    with open(checkpoints/'config.yaml', 'w') as config_file:
        yaml.dump(vars(args),
                  config_file,
                  default_flow_style = False)

    # model and loss function
    model = Model(prob = prob).to(device)
    loss_fn = nn.MSELoss()

    # optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # schedular
    milestones = range(num_warmup_epochs, num_epochs, sched_steps)
    scheduler = MultiStepLR(optimizer,
                            milestones = milestones,
                            gamma = sched_gamma)

    # data loader
    dataset_train = DatasetTPC(DATA_ROOT, split = 'train', dimension = 3)
    dataset_valid = DatasetTPC(DATA_ROOT, split = 'test',  dimension = 3)
    dataloader_train = DataLoader(dataset_train,
                                  batch_size = batch_size,
                                  shuffle = True)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size = batch_size)

    # get dummy data for scripting
    data = dataset_train[0]
    dummy_input = get_jit_input(data, batch_size, device)

    # get inference time
    if prob is None:
        samples_per_second = runtime(model.prob_predictor,
                                     input_shape = data.shape,
                                     batch_size = batch_size,
                                     num_inference_batches = 1000,
                                     script = True,
                                     device = device)
        print(f'samples per second = {samples_per_second: .1f}')

    ckpt_saver_prb = CheckpointSaver(checkpoints, save_frequency, prefix='prb')
    ckpt_saver_dec = CheckpointSaver(checkpoints, save_frequency, prefix='dec')

    # training
    df_data_train = defaultdict(list)
    df_data_valid = defaultdict(list)

    for epoch in range(1, num_epochs + 1):

        current_lr = get_lr(optimizer)
        print(f'current learning rate = {current_lr:.10f}')

        # train
        desc = f'Train Epoch {epoch} / {num_epochs}'
        train_stat = run_epoch(model,
                               loss_fn,
                               dataloader_train,
                               prob_lambda       = prob_lambda,
                               prob_lower_bound  = prob_lower_bound,
                               desc              = desc,
                               optimizer         = optimizer,
                               batches_per_epoch = batches_per_epoch,
                               device            = device)

        # validation
        with torch.no_grad():
            desc = f'Validation Epoch {epoch} / {num_epochs}'
            valid_stat = run_epoch(model,
                                   loss_fn,
                                   dataloader_valid,
                                   prob_lambda       = prob_lambda,
                                   prob_lower_bound  = prob_lower_bound,
                                   desc              = desc,
                                   batches_per_epoch = batches_per_epoch,
                                   device            = device)

        # save checkpoints
        ckpt_saver_prb(model.prob_predictor,
                       dummy_input,
                       epoch,
                       valid_stat['reco error'])
        ckpt_saver_dec(model.decoder,
                       dummy_input,
                       epoch,
                       valid_stat['reco error'])

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
        df_train.to_csv(checkpoints/'train_log.csv',
                        index = False,
                        float_format='%.6f')
        df_valid.to_csv(checkpoints/'valid_log.csv',
                        index = False,
                        float_format='%.6f')

        # update learning rate
        scheduler.step()

main()
