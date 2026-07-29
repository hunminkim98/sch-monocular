import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    """
    module의 parameter를 0으로 만들고 반환한다.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
