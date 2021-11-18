"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Copied from UNITER
(https://github.com/ChenRocks/UNITER)

Misc utilities
"""
import random

import torch
import numpy as np

from utils.logger import LOGGER


class Struct(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p:
                module.p = drop_p
                LOGGER.info(f'{name} set to {drop_p}')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TensorBackedImmutableStringArray:
    def __init__(self, strings, encoding = 'utf-8'):
        encoded = [torch.ByteTensor(torch.ByteStorage.from_buffer(s.encode(encoding))) for s in strings]
        self.cumlen = torch.cat((torch.zeros(1, dtype = torch.int64), torch.as_tensor(list(map(len, encoded)), dtype = torch.int64).cumsum(dim = 0)))
        self.data = torch.cat(encoded)
        self.encoding = encoding

    def __getitem__(self, i):
        if isinstance(i, slice):
            l = [self[idx] for idx in range(*i.indices(len(self)))]
            return TensorBackedImmutableStringArray(l)
        elif isinstance(i, int):
            return bytes(self.data[self.cumlen[i].data : self.cumlen[i + 1].data]).decode(self.encoding)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.cumlen) - 1

    def __list__(self):
        return [self[i] for i in range(len(self))]
