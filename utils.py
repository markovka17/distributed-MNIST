from logging import info

from torch.distributed import get_rank, get_world_size

def thread_info():
    print('^' * 30)
    print("{} / {} LOCAL RANK".format(get_rank() + 1, get_world_size()))
    print()

def prepare_batch(batch, device, non_blocking):
    x, y = batch
    x = x.cuda(device, non_blocking)
    y = y.cuda(device, non_blocking)

    return x, y

def save_thread(device_rank):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if get_rank() == device_rank:
                return func(*args, **kwargs)
            return
        return wrapper
    return decorator

if __name__ == "__main__":
    pass
