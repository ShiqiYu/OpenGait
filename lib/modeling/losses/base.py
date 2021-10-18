import torch.nn as nn
from utils import Odict

class BasicLoss(nn.Module):
    def __init__(self, loss_term_weights=1.0):
        super(BasicLoss, self).__init__()

        self.loss_term_weights = loss_term_weights
        self.pair_based_loss   = True
        self.info              = Odict()
    
    def forward(self, logits, labels):
        raise NotImplementedError
