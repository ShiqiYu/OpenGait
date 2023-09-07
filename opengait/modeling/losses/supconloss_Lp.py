from .base import BaseLoss, gather_and_scale_wrapper
from pytorch_metric_learning import losses, distances

class SupConLoss_Lp(BaseLoss):
    def __init__(self, temperature=0.01):
        super(SupConLoss_Lp, self).__init__()
        self.distance = distances.LpDistance()
        self.train_loss = losses.SupConLoss(temperature=temperature, distance=self.distance)
    @gather_and_scale_wrapper
    def forward(self, features, labels=None, mask=None):
        loss = self.train_loss(features,labels)
        self.info.update({
            'loss': loss.detach().clone()})
        return loss, self.info

