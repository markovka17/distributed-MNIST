import torch
from torch.distributed import get_rank, get_world_size
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms



def get_dataloaders(root):
    train_dataset = \
        datasets.MNIST(root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    train_loader = DataLoader(train_dataset, batch_size=32, drop_last=True)

    test_dataset = \
        datasets.MNIST(root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    test_loader = DataLoader(test_dataset, batch_size=32, drop_last=True)

    return train_loader, test_loader


def get_parallel_dataloaders(prefix, bz=32, nproc=1, pin_memory=True):
    bz_per_proc = int(bz / get_world_size())
    num_workers = int(nproc / get_world_size())

    train_root = '.data'
    train_dataset = \
        datasets.MNIST(
            train_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]
            )
        )

    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bz_per_proc, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory)

    test_root = train_root
    test_dataset = \
        datasets.MNIST(
            test_root, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]
            )
        )

    #test_sampler = DistributedSampler(
    #    test_dataset, num_replicas=get_world_size(), rank=get_rank())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=bz_per_proc,
        num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader
