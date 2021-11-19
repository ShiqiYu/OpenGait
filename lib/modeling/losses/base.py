import torch.nn as nn
from utils import Odict

class BaseLoss(nn.Module):
    def __init__(self, loss_term_weights=1.0):
        super(BaseLoss, self).__init__()

        self.loss_term_weights = loss_term_weights
        self.pair_based_loss   = True
        self.info              = Odict()
    
    def forward(self, logits, labels):
        raise NotImplementedError
