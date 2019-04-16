import os
import time
import argparse
import logging
from logging import info
# logging.basicConfig(level=logging.DEBUG)
from pprint import pprint

import torch
import torch.optim as optim
from torch.distributed import (
    init_process_group, get_rank, get_world_size)

from utils import prepare_batch, save_thread, thread_info
from model import Net
from data import get_parallel_dataloaders

import ignite
from ignite.handlers import Timer
from ignite.engine import (
    create_supervised_trainer, Events, create_supervised_evaluator)
from ignite.metrics import (
    Accuracy, Loss)

from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--logdir-suffix', type=str, default='mnist/dist')

    parser.add_argument('--mp-start-method', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--init-method', type=str,
                        default='file:///mnt/nfs/sharedfile')

    return parser.parse_args()


def main():
    """
    Distributed 4 GPUs
    """
    # torch.multiprocessing.set_start_method('spawn', force=True)


    args = parse_args()
    pprint(vars(args))

    if args.mp_start_method:
        torch.multiprocessing.set_start_method(args.mp_start_method)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda:{}'.format(args.local_rank))
    torch.cuda.set_device(device.index)

    print("Start initialization of process group...")
    init_process_group(args.backend, init_method=args.init_method,
                       rank=args.local_rank, world_size=args.world_size)

    thread_info()

    train_loader, test_loader = get_parallel_dataloaders('.data')

    model = Net().cuda(device)
    model = torch.nn.parallel.DistributedDataParallel(
        module=model, device_ids=[args.local_rank],
        output_device=args.local_rank)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.8)

    criterion = torch.nn.CrossEntropyLoss().cuda(device)

    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device, non_blocking=True,
        prepare_batch=prepare_batch)

    metrics = {'avg_accuracy': Accuracy(), 'avg_loss': Loss(criterion)}
    trainer_evaluator = create_supervised_evaluator(
        model, metrics, device=device, non_blocking=True,
        prepare_batch=prepare_batch)

    if get_rank() == 0:
        writer = SummaryWriter(
            os.path.join('/experiments',
                         args.logdir_suffix + str(get_world_size())))


    '''
    @trainer.on(Events.EPOCH_COMPLETED)
    @save_thread(device_rank=0)
    def log_training_loss(engine):
        metrics = trainer_evaluator.run(train_loader).metrics
        lr_scheduler.step()

        print("""Epoch[{}] Loss: {:.4f} Accuracy: {:.4f} LR: {:.4f} TIME: {}""" \
              .format(engine.state.epoch,
                      metrics['avg_loss'],
                      metrics['avg_accuracy'],
                      optimizer.param_groups[0]['lr'],
                      timer.value()))
    '''

    @trainer.on(Events.EPOCH_STARTED)
    def set_epoch(engine):
        train_loader.sampler.set_epoch(engine.state.epoch)

    @trainer.on(Events.ITERATION_COMPLETED)
    @save_thread(device_rank=0)
    def log_training_loss_iter(engine):
        writer.add_scalar(
            'loss', engine.state.output / args.batch_size,
            engine.state.iteration
        )


        iteration = (engine.state.iteration - 1) % len(train_loader) + 1
        if iteration % 600 == 0:

            print("Thread[{}/{}] Epoch[{}] Iteration[{}/{}] | Loss: {:.4f}" \
                  .format(get_rank() + 1, get_world_size(),
                          engine.state.epoch, iteration,
                          len(train_loader), engine.state.output))


    trainer.run(train_loader, max_epochs=5)

main()
