import torch
import torch.nn.functional as F
import numpy as np
from utils import ddp_all_gather
from .base import BaseLoss, gather_and_scale_wrapper


class InfoLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(InfoLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.filter = False
        self.temperature=0.1

    # @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
        labels = ddp_all_gather(labels)
        embeddings = ddp_all_gather(embeddings)
        # embeddings: [n, p, c], label: [n]
        
        bs = embeddings.shape[0]
        p = embeddings.shape[1]
        embeddings = embeddings.permute(
            1, 0, 2).contiguous()  # [n, p, c] -> [p, n, c]
        embeddings = embeddings.float()
        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]

        # embeddings_sim = embeddings.permute(0, 1, 2).reshape(bs, -1)
        # embeddings_sim =  F.normalize(embeddings, dim=-1)
        embeddings_sim =  embeddings
        # embeddings_sim =  F.normalize(embeddings_sim, dim=-1)
        sim_matrix = self.ComputeSim(embeddings_sim, embeddings_sim) # [1, n1, n2]
        # sim_matrix = sim_matrix.unsqueeze(0)
        positive_logit, negative_logits, ap_dist, an_dist = self.Convert(labels, ref_label, sim_matrix, dist)
        
        positive_logit = positive_logit.unsqueeze(2)
        negative_logits = negative_logits.unsqueeze(2)
        logits = torch.cat([positive_logit, negative_logits], dim=2)

        logits = logits.reshape(-1, logits.shape[-1])
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=embeddings.device)
        
        if self.filter:
            pick_id = np.random.randint(p)
            # mask = (logits[:, 0].detach()==torch.min(logits.detach(), dim=1)[0]) * 0.5 + 0.5
            # mask = (positive_logit.squeeze(2)[pick_id, :] > 0.5) * 0.5 + 0.5
            mask_logits = torch.cat([positive_logit, negative_logits], dim=2)
            mask = ~(torch.max(mask_logits[pick_id, ...], dim=1)[1].bool()) * 0.5 + 0.5
            mask_num = 0
            mask = mask.unsqueeze(0).repeat(p, 1).view(-1)
        else:
            mask = 1.
            mask_tri = 1.
            mask_num = 0
       
        info_loss = mask * F.cross_entropy(logits / self.temperature, labels, reduction='none')
        info_loss = info_loss.reshape(p, -1)
        
        loss_avg = info_loss.mean(dim=1)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'mask_num': mask_num,
            })

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeSim(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        # import pdb; pdb.set_trace()
        # sim_matrix = torch.einsum('nc,kc->nk', x, y)
        sim_matrix = x.matmul(y.transpose(-1, -2))
        return sim_matrix
    
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
    
    def Convert(self, row_labels, clo_label, sim_matrix, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).byte()  # [n_r, n_c]
        diffenc = matches ^ 1  # [n_r, n_c]
        mask = matches.unsqueeze(2) * diffenc.unsqueeze(1)
        a_idx, p_idx, n_idx = torch.where(mask)

        ap_sim_matrix = sim_matrix[:, a_idx, p_idx]
        an_sim_matrix = sim_matrix[:, a_idx, n_idx]

        ap_dist = dist[:, a_idx, p_idx]
        an_dist = dist[:, a_idx, n_idx]

        # print(sum(a_idx[10000:20000]==n_idx[10000:20000]))

        return ap_sim_matrix, an_sim_matrix, ap_dist, an_dist

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]
