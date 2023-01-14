"""The loss aggregator."""

import torch
import torch.nn as nn
from . import losses
from utils import is_dict, get_attr_from, get_valid_args, is_tensor, get_ddp_module
from utils import Odict
from utils import get_msg_mgr


class LossAggregator(nn.Module):
    """The loss aggregator.

    This class is used to aggregate the losses.
    For example, if you have two losses, one is triplet loss, the other is cross entropy loss,
    you can aggregate them as follows:
    loss_num = tripley_loss + cross_entropy_loss 

    Attributes:
        losses: A dict of losses.
    """
    def __init__(self, loss_cfg) -> None:
        """
        Initialize the loss aggregator.

        LossAggregator can be indexed like a regular Python dictionary, 
        but modules it contains are properly registered, and will be visible by all Module methods.
        All parameters registered in losses can be accessed by the method 'self.parameters()',
        thus they can be trained properly.
        
        Args:
            loss_cfg: Config of losses. List for multiple losses.
        """
        super().__init__()
        self.losses = nn.ModuleDict({loss_cfg['log_prefix']: self._build_loss_(loss_cfg)} if is_dict(loss_cfg) \
            else {cfg['log_prefix']: self._build_loss_(cfg) for cfg in loss_cfg})

    def _build_loss_(self, loss_cfg):
        """Build the losses from loss_cfg.

        Args:
            loss_cfg: Config of loss.
        """
        Loss = get_attr_from([losses], loss_cfg['type'])
        valid_loss_arg = get_valid_args(
            Loss, loss_cfg, ['type', 'gather_and_scale'])
        loss = get_ddp_module(Loss(**valid_loss_arg).cuda())
        return loss

    def forward(self, training_feats):
        """Compute the sum of all losses.

        The input is a dict of features. The key is the name of loss and the value is the feature and label. If the key not in 
        built losses and the value is torch.Tensor, then it is the computed loss to be added loss_sum.

        Args:
            training_feats: A dict of features. The same as the output["training_feat"] of the model.
        """
        loss_sum = .0
        loss_info = Odict()

        for k, v in training_feats.items():
            if k in self.losses:
                loss_func = self.losses[k]
                loss, info = loss_func(**v)
                for name, value in info.items():
                    loss_info['scalar/%s/%s' % (k, name)] = value
                loss = loss.mean() * loss_func.loss_term_weight
                loss_sum += loss

            else:
                if isinstance(v, dict):
                    raise ValueError(
                        "The key %s in -Trainng-Feat- should be stated in your loss_cfg as log_prefix."%k
                    )
                elif is_tensor(v):
                    _ = v.mean()
                    loss_info['scalar/%s' % k] = _
                    loss_sum += _
                    get_msg_mgr().log_debug(
                        "Please check whether %s needed in training." % k)
                else:
                    raise ValueError(
                        "Error type for -Trainng-Feat-, supported: A feature dict or loss tensor.")

        return loss_sum, loss_info
