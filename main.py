import time
import argparse

import torch
import torch.optim as optim
from torch.distributed import init_process_group
import torch.distributed as distr

from utils import prepare_batch
from model import Net
from data import get_dataloaders

from ignite.handlers import Timer
from ignite.engine import (
    create_supervised_trainer, Events, create_supervised_evaluator)
from ignite.metrics import (
    Accuracy, Loss)

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M')

    parser.add_argument('--local-rank', type=int, default=0)

    return parser.parse_args()


def main():
    """
    Single GPU
    """

    args = parse_args()

    device = torch.device('cuda:{}'.format(args.local_rank))

    writer = SummaryWriter('/experiments/mnist/single_gpu')

    train_loader, test_loader = get_dataloaders('.data')
    model = Net().cuda(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.8)

    criterion = torch.nn.CrossEntropyLoss().cuda(device)

    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device, non_blocking=False,
        prepare_batch=prepare_batch
    )
    metrics = {'avg_accuracy': Accuracy(), 'avg_loss': Loss(criterion)}
    trainer_evaluator = create_supervised_evaluator(
        model, metrics, device=device, non_blocking=False,
        prepare_batch=prepare_batch)


    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        pause=Events.EPOCH_COMPLETED,
        resume=Events.EPOCH_STARTED,
        step=Events.EPOCH_COMPLETED
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(engine):
        metrics = trainer_evaluator.run(train_loader).metrics
        lr_scheduler.step()

        print("""Epoch[{}] Loss: {:.4f} Accuracy: {:.4f} LR: {:.4f} TIME: {}""" \
              .format(engine.state.epoch,
                      metrics['avg_loss'],
                      metrics['avg_accuracy'],
                      optimizer.param_groups[0]['lr'],
                      timer.value()))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss_iter(engine):
        writer.add_scalar(
            'loss', engine.state.output / args.batch_size,
            engine.state.iteration
        )


        iteration = (engine.state.iteration - 1) % len(train_loader) + 1
        if iteration % 600 == 0:

            print("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}" \
                  .format(engine.state.epoch, iteration, len(train_loader),
                          engine.state.output))


    trainer.run(train_loader, max_epochs=5)


main()
