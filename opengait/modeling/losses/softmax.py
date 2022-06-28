import torch
import torch.nn.functional as F

from .base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

    def forward(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.size()
        log_preds = F.log_softmax(logits * self.scale, dim=1)  # [n, c, p]
        one_hot_labels = self.label2one_hot(
            labels, c).unsqueeze(2).repeat(1, 1, p)  # [n, c, p]
        loss = self.compute_loss(log_preds, one_hot_labels)
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels.unsqueeze(1)).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info

    def compute_loss(self, predis, labels):
        softmax_loss = -(labels * predis).sum(1)  # [n, p]
        losses = softmax_loss.mean(0)   # [p]

        if self.label_smooth:
            smooth_loss = - predis.mean(dim=1)  # [n, p]
            smooth_loss = smooth_loss.mean(0)  # [p]
            losses = smooth_loss * self.eps + losses * (1. - self.eps)
        return losses

    def label2one_hot(self, label, class_num):
        label = label.unsqueeze(-1)
        batch_size = label.size(0)
        device = label.device
        return torch.zeros(batch_size, class_num).to(device).scatter(1, label, 1)
