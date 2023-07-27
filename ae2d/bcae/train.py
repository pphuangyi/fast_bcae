"""
"""
from collections import defaultdict
import argparse
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import yaml

import torch
from torch import nn
from torch.nn.functional import pad
from torch.jit import trace, save
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

from dataset import DatasetTPC2d
from networks import Encoder, Decoder
from loss import BCAELoss
# from runtime import runtime


MANIFEST_TRAIN = '/data/datasets/sphenix/highest_framedata_3d/outer/train.txt'
MANIFEST_VALID = '/data/datasets/sphenix/highest_framedata_3d/outer/test.txt'

def get_args(description):
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description)

    parser.add_argument('--clamp',
                        action = 'store_true',
                        help   = ('if used, clamp with TPC reange (64, 1023]'))
    parser.add_argument('--clf-lambda',
                        dest = 'clf_lambda',
                        type = float,
                        default = 1.,
                        help   = ('Coefficient for classification loss '
                                  '(default = 1.)'))
    parser.add_argument('--clf-threshold',
                        dest = 'clf_threshold',
                        type = float,
                        default = 0.5,
                        help   = ('Threshold for classification output '
                                  '(default = 0.5)'))
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
                        default = .95,
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
              clamp = False):
    """
    """
    total = min(batches_per_epoch, len(dataloader))
    pbar = tqdm(desc = desc, total = total)

    loss_sum = defaultdict(float)
    true, pos, true_pos = 0, 0, 0

    for idx, adc in enumerate(dataloader):

        if idx >= batches_per_epoch:
            break

        # pad the z dimension to have length 256
        tag = adc > 64

        tag = tag.to(device)
        adc = adc.to(device)

        code = encoder(pad(adc, (0, 7)))
        clf_output, reg_output = decoder(code)
        clf_output = clf_output[..., :-7]
        reg_output = reg_output[..., :-7]
        reg_output = torch.clamp(reg_output, max=5)

        results = loss_fn(clf_output, reg_output, tag, adc)
        loss = results.pop('loss')

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if clamp:
        #     # output = nn.functional.relu(output, inplace=True)
        #     output = torch.exp(output) * 6 + 64

        true += results.pop('true')
        pos += results.pop('pos')
        true_pos += results.pop('true pos')
        clf_coef = results.pop('clf coef')

        loss_sum['loss'] += loss.item()
        for key, val in results.items():
            loss_sum[key] += val

        pbar.update()
        postfix = {key: val / (idx + 1) for
                   key, val in loss_sum.items()}
        postfix['precision'] = true_pos / pos
        postfix['recall'] = true_pos / true
        postfix['clf coef'] = clf_coef
        pbar.set_postfix(postfix)

    return postfix


class BiDecoder(nn.Module):
    def __init__(self, in_channels, num_blocks, num_downsamples):
        super().__init__()
        self.decoder_clf = Decoder(in_channels,
                                   num_blocks,
                                   num_downsamples,
                                   output_activ = 'sigmoid')
        self.decoder_reg = Decoder(in_channels,
                                   num_blocks,
                                   num_downsamples)

    def forward(self, data):
        output_clf = self.decoder_clf(data)
        output_reg = self.decoder_reg(data)
        return output_clf, output_reg


def main():

    args = get_args('2d TPC Data Compression')

    clamp             = args.clamp
    clf_lambda        = args.clf_lambda
    clf_threshold     = args.clf_threshold
    num_epochs        = args.num_epochs
    num_warmup_epochs = args.num_warmup_epochs
    batch_size        = args.batch_size
    batches_per_epoch = args.batches_per_epoch
    sched_steps       = args.sched_steps
    sched_gamma       = args.sched_gamma
    device            = args.device
    learning_rate     = args.learning_rate
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
    encoder = Encoder(16, 5, 3).to(device)
    decoder = BiDecoder(16, 5, 3).to(device)
    transform = lambda x : torch.exp(x) * 6 + 64.
    # transform = None
    loss_fn = BCAELoss(transform, clf_threshold, weight_pow = None)

    # optimizer
    params = chain(encoder.parameters(), decoder.parameters())
    optimizer = AdamW(params, lr=learning_rate)

    # schedular
    milestones = range(num_warmup_epochs, num_epochs, sched_steps)
    scheduler = MultiStepLR(optimizer,
                            milestones = milestones,
                            gamma = sched_gamma)

    # data loader
    dataset_train = DatasetTPC2d(MANIFEST_TRAIN)
    dataset_valid = DatasetTPC2d(MANIFEST_VALID)
    dataloader_train = DataLoader(dataset_train,
                                  batch_size = batch_size,
                                  shuffle = True)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size = batch_size)

    # get dummy data for scripting
    adc = dataset_train[0]
    dummy_input = get_jit_input(adc, batch_size, device)
    with torch.no_grad():
        dummy_compr = encoder(dummy_input)

    # # get inference time
    # samples_per_second = runtime(encoder,
    #                              input_shape = data.shape,
    #                              batch_size = batch_size,
    #                              num_inference_batches = 1000,
    #                              script = True,
    #                              device = device)
    # print(f'samples per second = {samples_per_second: .1f}')

    ckpt_saver_enc = CheckpointSaver(checkpoints, save_frequency, prefix='enc')
    ckpt_saver_dec = CheckpointSaver(checkpoints, save_frequency, prefix='dec')

    for epoch in range(1, num_epochs + 1):

        # train
        desc = f'Train Epoch {epoch} / {num_epochs}'
        train_stat = run_epoch(encoder,
                               decoder,
                               loss_fn,
                               dataloader_train,
                               desc = desc,
                               optimizer = optimizer,
                               batches_per_epoch = batches_per_epoch,
                               device = device,
                               clamp = clamp)

        # validation
        with torch.no_grad():
            desc = f'Validation Epoch {epoch} / {num_epochs}'
            valid_stat = run_epoch(encoder,
                                   decoder,
                                   loss_fn,
                                   dataloader_valid,
                                   desc = desc,
                                   batches_per_epoch = batches_per_epoch,
                                   device = device,
                                   clamp = clamp)

        # save checkpoints
        ckpt_saver_enc(encoder,
                       dummy_input,
                       epoch,
                       valid_stat['mse'])
        ckpt_saver_dec(decoder,
                       dummy_compr,
                       epoch,
                       valid_stat['mse'])

        # update learning rate
        scheduler.step()
        current_lr = get_lr(optimizer)
        print(f'current learning rate = {current_lr:.10f}')

main()
