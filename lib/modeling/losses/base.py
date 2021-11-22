from ctypes import ArgumentError
import torch.nn as nn
from utils import Odict
import functools
from utils import ddp_all_gather


def gather_input(func):
    """Internal wrapper: gather the input from multple cards to one card.
    """

    @functools.wraps(func)
    def inner(*args, **kwds):
        try:
            if args[0].pair_based_loss:
                for k, v in kwds.items():
                    kwds[k] = ddp_all_gather(v)
        except:
            raise ArgumentError
        return func(*args, **kwds)
    return inner


class BaseLoss(nn.Module):
    """
    Base class for all losses.

    Your loss should also subclass this class. 

    Attribute:
        loss_term_weights: the weight of the loss.
        pair_based_loss: indicates whether the loss needs to make pairs like the triplet loss.
        info: the loss info.
    """
    loss_term_weights = 1.0
    pair_based_loss = False
    info = Odict()

    @gather_input
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
        loss = .0
        return loss, self.info
