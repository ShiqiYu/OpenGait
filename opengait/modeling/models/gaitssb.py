import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import PackSequenceWrapper, HorizontalPoolingPyramid, SetBlockWrapper, ParallelBN1d, SeparateFCs

from utils import np2var, list2var, get_valid_args, ddp_all_gather
from data.transform import get_transform
from einops import rearrange

# Modified from https://github.com/PatrickHua/SimSiam/blob/main/models/simsiam.py
class GaitSSB_Pretrain(BaseModel):
    def __init__(self, cfgs, training=True):
        super(GaitSSB_Pretrain, self).__init__(cfgs, training=training)

    def build_network(self, model_cfg):
        self.p = model_cfg['parts_num']
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid([16, 8, 4, 2, 1])

        out_channels = model_cfg['backbone_cfg']['channels'][-1]
        hidden_dim = out_channels
        self.projector = nn.Sequential(SeparateFCs(self.p, out_channels, hidden_dim),
                                ParallelBN1d(self.p, hidden_dim),
                                nn.ReLU(inplace=True),
                                SeparateFCs(self.p, hidden_dim, out_channels),
                                ParallelBN1d(self.p, out_channels))
        self.predictor = nn.Sequential(SeparateFCs(self.p, out_channels, hidden_dim),
                                ParallelBN1d(self.p, hidden_dim),
                                nn.ReLU(inplace=True),
                                SeparateFCs(self.p, hidden_dim, out_channels))

    def inputs_pretreament(self, inputs):
        if self.training:
            seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
            trf_cfgs = self.engine_cfg['transform']
            seq_trfs = get_transform(trf_cfgs)

            requires_grad = True if self.training else False
            batch_size = int(len(seqs_batch[0]) / 2)
            img_q = [np2var(np.asarray([trf(fra) for fra in seq[:batch_size]]), requires_grad=requires_grad).float()  for trf, seq in zip(seq_trfs, seqs_batch)]
            img_k = [np2var(np.asarray([trf(fra) for fra in seq[batch_size:]]), requires_grad=requires_grad).float()  for trf, seq in zip(seq_trfs, seqs_batch)]
            seqs = [img_q, img_k]

            typs = typs_batch
            vies = vies_batch

            if self.training:
                labs = list2var(labs_batch).long()
            else:
                labs = None

            if seqL_batch is not None:
                seqL_batch = np2var(seqL_batch).int()
            seqL = seqL_batch

            ipts = seqs
            del seqs

            return ipts, labs, typs, vies, (seqL, seqL)
        else:
            return super().inputs_pretreament(inputs)

    def encoder(self, inputs):
        sils, seqL = inputs
        assert sils.size(-1) in [44, 88]
        outs = self.Backbone(sils) # [n, c, s, h, w]
        outs = self.TP(outs, seqL, options={"dim": 2})[0] # [n, c, h, w]
        feat = self.HPP(outs) # [n, c, p], Horizontal Pooling, HP
        return feat

    def forward(self, inputs):
        '''
        Input: 
            sils_q: a batch of query images, [n, s, h, w]
            sils_k: a batch of key images, [n, s, h, w]
        Output:
            logits, targets
        '''
        if self.training:
            (sils_q, sils_k), labs, typs, vies, (seqL_q, seqL_k) = inputs

            sils_q, sils_k = sils_q[0].unsqueeze(1), sils_k[0].unsqueeze(1)

            q_input = (sils_q, seqL_q)
            q_feat = self.encoder(q_input) # [n, c, p]
            z1 = self.projector(q_feat)
            p1 = self.predictor(z1)

            k_input = (sils_k, seqL_k)
            k_feat = self.encoder(k_input) # [n, c, p]
            z2 = self.projector(k_feat)
            p2 = self.predictor(z2)

            logits1, labels1 = self.D(p1, z2)
            logits2, labels2 = self.D(p2, z1)

            retval = {
                    'training_feat': {'softmax1': {'logits': logits1, 'labels': labels1},
                        'softmax2': {'logits': logits2, 'labels': labels2}
                    },
                    'visual_summary': {'image/encoder_q': rearrange(sils_q, 'n c s h w -> (n s) c h w'),
                        'image/encoder_k': rearrange(sils_k, 'n c s h w -> (n s) c h w'),
                    },
                    'inference_feat': None
            }
            return retval
        else:
            sils, labs, typs, vies, seqL = inputs
            sils = sils[0].unsqueeze(1)
            feat = self.encoder((sils, seqL)) # [n, c, p]
            feat = self.projector(feat) # [n, c, p]
            feat = self.predictor(feat) # [n, c, p]
            retval = {
                'training_feat': None,
                'visual_summary': None,
                'inference_feat': {'embeddings': F.normalize(feat, dim=1)}
            }
            return retval

    def D(self, p, z): # negative cosine similarity
        """
            p: [n, c, p]
            z: [n, c, p]
        """
        z = z.detach() # stop gradient
        n = p.size(0)

        p = F.normalize(p, dim=1) # l2-normalize, [n, c, p]
        z = F.normalize(z, dim=1) # l2-normalize, [n, c, p]
        z = ddp_all_gather(z, dim=0, requires_grad=False) # [m, c, p],  m = n * the number of GPUs

        logits = torch.einsum('ncp, mcp->nmp', [p, z]) # [n, m, p]
        rank   = torch.distributed.get_rank()
        labels = torch.arange(rank*n, (rank+1)*n, dtype=torch.long).cuda()
        return logits, labels

import torch.optim as optim
import numpy as np
from utils import get_valid_args, list2var

class no_grad(torch.no_grad):
    def __init__(self, enable=True):
        super(no_grad, self).__init__()
        self.enable = enable

    def __enter__(self):
        if self.enable:
            super().__enter__()
        else:
            pass

    def __exit__(self, *args):
        if self.enable:
            super().__exit__(*args)
        else:
            pass

class GaitSSB_Finetune(BaseModel):
    def __init__(self, cfgs, training=True):
        super(GaitSSB_Finetune, self).__init__(cfgs, training=training)

    def build_network(self, model_cfg):
        self.p = model_cfg['parts_num']
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid([16, 8, 4, 2, 1])

        out_channels = model_cfg['backbone_cfg']['channels'][-1]
        hidden_dim = out_channels
        self.projector = nn.Sequential(SeparateFCs(self.p, out_channels, hidden_dim), 
                                ParallelBN1d(self.p, hidden_dim), 
                                nn.ReLU(inplace=True), 
                                SeparateFCs(self.p, hidden_dim, out_channels), 
                                ParallelBN1d(self.p, out_channels))

        self.backbone_lr  = model_cfg['backbone_lr']
        self.projector_lr = model_cfg['projector_lr']

        self.head0 = SeparateFCs(self.p, out_channels, out_channels, norm=True)

    def get_optimizer(self, optimizer_cfg):
        optimizer = getattr(optim, optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])

        ft_param_list  = []
        self.fix_layer = []
        for i, ft_lr in enumerate(self.backbone_lr):
            if ft_lr != 0:
                ft_param_list.append({
                    'params': getattr(self.Backbone.forward_block, 'layer%d'%(i+1)).parameters(), 
                    'lr': ft_lr, 
                })
            else:
                self.fix_layer.append('layer%d'%(i+1))

        ft_param_list.append({
            'params': self.projector.parameters(), 
            'lr': self.projector_lr, 
        })

        ft_param_list.append({
            'params': self.head0.parameters(), 
            'lr': valid_arg['lr']
        })

        optimizer = optimizer(ft_param_list, **valid_arg)

        return optimizer

    def encoder(self, inputs):
        sils, seqL = inputs
        n = sils.size(0)
        sils = rearrange(sils, 'n c s h w -> (n s) c h w')

        if not self.training:
            self.fix_layer = ['layer1', 'layer2', 'layer3', 'layer4']

        with no_grad(): 
            outs = self.Backbone.forward_block.conv1(sils)
            outs = self.Backbone.forward_block.bn1(outs)
            outs = self.Backbone.forward_block.relu(outs)

        with no_grad('layer1' in self.fix_layer):
            outs = self.Backbone.forward_block.layer1(outs)

        with no_grad('layer2' in self.fix_layer):
            outs = self.Backbone.forward_block.layer2(outs)

        with no_grad('layer3' in self.fix_layer):
            outs = self.Backbone.forward_block.layer3(outs)

        with no_grad('layer4' in self.fix_layer):
            outs = self.Backbone.forward_block.layer4(outs)

        outs = rearrange(outs, '(n s) c h w -> n c s h w', n=n)
        outs = self.TP(outs, seqL, options={"dim": 2})[0] # [n, c, h, w]

        feat = self.HPP(outs) # [n, c, p], Horizontal Pooling, HP
        return feat

    def forward(self, inputs):
        if self.training:
            self.maintain_non_zero_learning_rate()

        sils, labs, typs, vies, seqL = inputs
        sils = sils[0].unsqueeze(1)
        feat = self.encoder([sils, seqL]) # [n, c, p]

        feat = self.projector(feat) # [n, c, p]
        feat = F.normalize(feat, dim=1)

        embed = self.head0(feat) # [n, c, p]

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval

    def maintain_non_zero_learning_rate(self):
        if self.iteration % 1000 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] < 1e-4:
                    param_group['lr'] = 1e-4