import torch
import torch.nn.functional as F

from .base import BasicLoss


class CrossEntropyLoss(BasicLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weights=1.0, log_accuracy=False):
        super(CrossEntropyLoss, self).__init__()
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

        self.loss_term_weights = loss_term_weights
        self.pair_based_loss = False

    def forward(self, logits, labels):
        """
            logits: [n, p, c]
            labels: [n]
        """
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p, c] -> [p, n, c]
        p, _, c = logits.size()
        log_preds = F.log_softmax(logits * self.scale, dim=-1)  # [p, n, c]
        one_hot_labels = self.label2one_hot(
            labels, c).unsqueeze(0).repeat(p, 1, 1)  # [p, n, c]
        loss = self.compute_loss(log_preds, one_hot_labels)
        self.info.update({'loss': loss})
        if self.log_accuracy:
            pred = logits.argmax(dim=-1)  # [p, n]
            accu = (pred == labels.unsqueeze(0)).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info

    def compute_loss(self, predis, labels):
        softmax_loss = -(labels * predis).sum(-1)  # [p, n]
        losses = softmax_loss.mean(-1)

        if self.label_smooth:
            smooth_loss = - predis.mean(dim=-1)  # [p, n]
            smooth_loss = smooth_loss.mean()  # [p]
            smooth_loss = smooth_loss * self.eps
            losses = smooth_loss + losses * (1. - self.eps)
        return losses

    def label2one_hot(self, label, class_num):
        label = label.unsqueeze(-1)
        batch_size = label.size(0)
        device = label.device
        return torch.zeros(batch_size, class_num).to(device).scatter(1, label, 1)
