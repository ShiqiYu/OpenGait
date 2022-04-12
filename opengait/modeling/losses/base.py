from ctypes import ArgumentError
import torch.nn as nn
import torch
from utils import Odict
import functools
from utils import ddp_all_gather


def gather_and_scale_wrapper(func):
    """Internal wrapper: gather the input from multple cards to one card, and scale the loss by the number of cards.
    """

    @functools.wraps(func)
    def inner(*args, **kwds):
        try:

            for k, v in kwds.items():
                kwds[k] = ddp_all_gather(v)

            loss, loss_info = func(*args, **kwds)
            loss *= torch.distributed.get_world_size()
            return loss, loss_info
        except:
            raise ArgumentError
    return inner


class BaseLoss(nn.Module):
    """
    Base class for all losses.

    Your loss should also subclass this class.
    """

    def __init__(self, loss_term_weight=1.0):
        """
        Initialize the base class.

        Args:
            loss_term_weight: the weight of the loss term.
        """
        super(BaseLoss, self).__init__()
        self.loss_term_weight = loss_term_weight
        self.info = Odict()

    def forward(self, logits, labels):
        """
        The default forward function.

        This function should be overridden by the subclass. 

        Args:
            logits: the logits of the model.
            labels: the labels of the data.

        Returns:
            tuple of loss and info.
        """
        return .0, self.info
