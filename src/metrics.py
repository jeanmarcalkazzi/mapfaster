import math

import torch
from fastai.vision.all import Metric, flatten_check

# fmt: off

def mapfast_accuracy(inp, *targ, axis=-1):
    """
    Compute accuracy with `targ` when `pred` is bs * n_classes
    Assume class is always returned as the first element of the tuple
    """
    pred, targ = flatten_check(inp.argmax(dim=axis), targ[0])
    return (pred == targ).float().mean()


def mapfast_coverage(inp, *targ, axis=-1):
    runtime_per_instance = torch.index_select(torch.stack(targ[4:]).cuda(), 0, inp.argmax(axis=axis).cuda()).diag().float()
    
    return torch.where(runtime_per_instance > 0, 1, 0).float().mean()


def mapfast_score(inp, *targ, axis=-1):
    # Get runtime of current model
    runtime_current_model = torch.index_select(torch.stack(targ[4:]).cuda(), 0, inp.argmax(axis=axis).cuda()).diag().float()
    # Get runtime per instance.  Instances are on axis=1
    runtime_per_instance_per_alg = torch.stack((*targ[4:], runtime_current_model)).cuda().double()
    # For each algorithm with unsolved instance, set it to +inf to get 0 speedFactor
    runtime_per_instance_per_alg = torch.where(runtime_per_instance_per_alg == -1, math.inf, runtime_per_instance_per_alg)
    # Calculate speed factor per algorithm per instance
    speed_factor = (300 / (1 + runtime_per_instance_per_alg)).float()

    score = (speed_factor / speed_factor.sum(dim=0))[-1].sum()

    return score


def mapfast_accuracy_per_maptype(inp, *targ, vocab=None, axis=-1):
    maps_accuracy = {}

    for i in torch.unique(targ[3]):
        _inp = inp[torch.where(targ[3] == i)]
        _targ = targ[0][torch.where(targ[3] == i)]
        p, t = flatten_check(_inp.argmax(dim=axis), _targ)
        if vocab is not None:
            maps_accuracy[vocab[i.item()]] = (p == t).float().mean().item()
        else:
            maps_accuracy[i.item()] = (p == t).float().mean().item()
    return maps_accuracy


def mapfast_runtime(inp, *targ, axis=-1):
    runtime_per_instance = torch.index_select(torch.stack(targ[4:]).cuda(), 0, inp.argmax(axis=axis).cuda()).diag().double()
    runtime_per_instance = torch.where(runtime_per_instance == -1, 300.0, runtime_per_instance)
    return runtime_per_instance.float().sum()


class SumMetric(Metric):
    def __init__(self, func):
        self.total = 0
        self.func = func

    def reset(self):
        self.total = 0

    def accumulate(self, learn):
        self.total += learn.to_detach(self.func(learn.pred, *learn.yb))

    @property
    def value(self):
        return self.total

    @property
    def name(self):
        return (
            self.func.func.__name__
            if hasattr(self.func, "func")
            else self.func.__name__
        )
