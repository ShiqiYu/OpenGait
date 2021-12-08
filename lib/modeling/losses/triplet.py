import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weights=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

        self.loss_term_weights = loss_term_weights

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, p, c], label: [n]
        embeddings = embeddings.permute(
            1, 0, 2).contiguous()  # [n, p, c] -> [p, n, c]
        embeddings = embeddings.float()

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean(1).mean(1)
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        dist_diff = ap_dist - an_dist
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(-1, -2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).byte()  # [n_r, n_c]
        diffenc = matches ^ 1  # [n_r, n_c]
        mask = matches.unsqueeze(2) * diffenc.unsqueeze(1)
        a_idx, p_idx, n_idx = torch.where(mask)

        ap_dist = dist[:, a_idx, p_idx]
        an_dist = dist[:, a_idx, n_idx]
        return ap_dist, an_dist
