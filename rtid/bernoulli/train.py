"""
"""
from collections import defaultdict
from tqdm import tqdm

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


def run_epoch(model,
              loss_fn,
              dataloader, *,
              prob_lambda       = 1000,
              prob_lower        = .1,
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
        prob_coef = prob_lambda * max(0., prob_loss.item() - prob_lower)
        loss = prob_loss + prob_coef * prob_loss

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

        total['gate'] += gate.item()
        total['gate_hard'] += gate_hard.item()
        total['reco_hard'] += loss_fn(reco_hard, adc).item()
        total['signal'] += true.item()

        # update the porgress bar
        pbar.update()
        postfix = {'loss': total['loss'] / idx,
                   'reco loss': total['reco_loss'] / idx,
                   'prob loss': total['prob_loss'] / idx,
                   'keep ratio': total['gate'] / total['signal'],
                   'reco error': total['reco_hard'] / idx,
                   'keep ratio hard': total['gate_hard'] / total['signal']}
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
    prob_lambda = 1000
    prob_lower = .1
    save_frequency = 10
    checkpoints = Path('./checkpoints')
    checkpoints.mkdir(exist_ok = True, parents = True)


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
    dataset_train = DatasetTPC(MANIFEST_TRAIN, dimension = 3)
    dataset_valid = DatasetTPC(MANIFEST_VALID, dimension = 3)
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

    ckpt_saver_enc = CheckpointSaver(checkpoints, save_frequency, prefix='enc')
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
                               prob_lower        = prob_lower,
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
                                   prob_lower        = prob_lower,
                                   desc              = desc,
                                   batches_per_epoch = batches_per_epoch,
                                   device            = device)

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
