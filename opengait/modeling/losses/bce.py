import torch
from .base import BaseLoss
from evaluation import mean_iou


class BinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0, eps=1.0e-9):
        super(BinaryCrossEntropyLoss, self).__init__(loss_term_weight)
        self.eps = eps

    def forward(self, logits, labels):
        """
            logits: [n, 1, h, w]
            labels: [n, 1, h, w]
        """
        # predts = torch.sigmoid(logits.float())
        labels = labels.float()
        logits = logits.float()

        loss = - (labels * torch.log(logits + self.eps) +
                  (1 - labels) * torch.log(1. - logits + self.eps))

        n = loss.size(0)
        loss = loss.view(n, -1)
        mean_loss = loss.mean()
        hard_loss = loss.max()
        miou = mean_iou((logits > 0.5).float(), labels)
        self.info.update({
            'loss': mean_loss.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'miou': miou.detach().clone()})

        return mean_loss, self.info


if __name__ == "__main__":
    loss_func = BinaryCrossEntropyLoss()
    ipts = torch.randn(1, 1, 128, 64)
    tags = (torch.randn(1, 1, 128, 64) > 0.).float()
    loss = loss_func(ipts, tags)
    print(loss)
