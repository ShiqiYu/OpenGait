import torch
import torch.nn as nn

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch.nn.functional as F

from ..base_model import BaseModel
from ..multidataset_model import MultiDatasets
from ..modules import ParallelBN1d, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
from utils import np2var, list2var, get_valid_args, ddp_all_gather
from data.transform import get_transform
from einops import rearrange


from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
import copy
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule
import torch.utils.checkpoint as cp
import torch.optim as optim

def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def freeze_eval_module(module: nn.Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


blocks_map = {
    '2d': BasicBlock2D, 
    'p3d': BasicBlockP3D, 
    '3d': BasicBlock3D
}

class DeepGaitV2_SSL(nn.Module):

    def __init__(self, model_cfg):
        super(DeepGaitV2_SSL, self).__init__()
        
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        if mode == '3d': 
            strides = [
                [1, 1], 
                [1, 2, 2], 
                [1, 2, 2], 
                [1, 1, 1]
            ]
        else: 
            strides = [
                [1, 1], 
                [2, 2], 
                [2, 2], 
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1), 
            nn.BatchNorm2d(self.inplanes), 
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))
        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d': 
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.p = model_cfg['parts_num']
        out_channels = model_cfg['Backbone']['channels'][-1]
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

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion

        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def encoder(self, sils):
        # assert sils.size(-1) in [44, 64, 88]
        out0 = self.layer0(sils) # [n,64,s,64,44]
        out1 = self.layer1(out0) # [n,64,s,64,44]
        out2 = self.layer2(out1) # [n,128,s,32,22]
        out3 = self.layer3(out2) # [n,256,s,16,11]
        outs = self.layer4(out3) # [n, c, s, h, w]
        return [out0, out1, out2, out3, outs]

    def forward(self, inputs):
        sils, seqL = inputs
        feat = self.encoder(sils)[-1] # [n, c, p]
        feat_tp = self.TP(feat, None, options={"dim": 2})[0] # [n, c, h, w]
        feat_hpp = self.HPP(feat_tp) # [n, c, p], Horizontal Pooling, HP
        z1 = self.projector(feat_hpp)
        p1 = self.predictor(z1)
        return feat, feat_tp, feat_hpp, z1, p1


class FoundationGait_Scaling(MultiDatasets):
    def __init__(self, cfgs, training):
        super(FoundationGait_Scaling, self).__init__(cfgs, training=training)

    def inputs_pretreament(self, inputs):
        import torch.autograd as autograd
        if self.training:
            seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
            trf_cfgs = self.engine_cfg['transform']

            requires_grad = True if self.training else False
            batch_size = int(len(seqs_batch[0]) / 2)

            seq_trfs_1, seq_trfs_2 = [get_transform(trf_cfgs[0][0])], [get_transform(trf_cfgs[0][1])]
            img_q = [np2var(np.asarray([trf(fra) for fra in seq[:batch_size]]), requires_grad=requires_grad).float()  for trf, seq in zip(seq_trfs_1, seqs_batch)]
            img_k = [np2var(np.asarray([trf(fra) for fra in seq[batch_size:]]), requires_grad=requires_grad).float()  for trf, seq in zip(seq_trfs_2, seqs_batch)]
            seqs = [img_q, img_k] # [teacher_global, student_global]

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

    def train_step(self, loss_sum) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """

        self.optimizer.zero_grad()
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        if self.engine_cfg['enable_float16']:
            self.Scaler.scale(loss_sum).backward()
            self.Scaler.step(self.optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            # Warning caused by optimizer skip when NaN
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/5
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug("Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                    scale, self.Scaler.get_scale()))
                return False
        else:
            loss_sum.backward()
            self.optimizer.step()

        self.iteration += 1
        self.scheduler.step()

        # Update Teacher EMA
        momentum = cosine_schedule(
            step=self.iteration,
            max_steps=self.engine_cfg['total_iter'],
            start_value=self.cfgs['model_cfg']['EMA_value'],
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        return True

    def build_network(self, model_cfg):
        self.teacher_backbone = DeepGaitV2_SSL(model_cfg)
        super().init_parameters() # init teacher_backbone
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        freeze_eval_module(self.teacher_backbone)
        self.crop_list = model_cfg['crop_list']
        pass

    def init_parameters(self):
        # Don't init agian!!! Keep teacher == student at begining.
        pass
        n_parameters = sum(p.numel() for p in self.teacher_backbone.parameters())
        self.msg_mgr.log_info('Teacher Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('All Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
        self.msg_mgr.log_info("=> init successfully")

    def diverse_crop_encode(self, x, num_part):
        n,c,s,h,w = x.shape
        assert n % 2 == 0
        if num_part == 1:
            return self.student_backbone.encoder(x)[-1]
        win_size = h//num_part
        x1, x2 = x[:n//2], x[n//2:]
        x1 = rearrange(x1, 'n c s (n_h w_h) w -> (n n_h) c s w_h w',n_h=num_part)
        x2 = F.pad(x2, (0, 0, win_size//2, win_size//2))
        x2 = x2.unfold(dimension=-2, size=win_size, step=win_size)
        x2 = rearrange(x2, 'n c s n_h w w_h -> (n n_h) c s w_h w',n_h=(num_part+1))
        feat = self.student_backbone.encoder(torch.concat([x1,x2]))[-1]
        feat1, feat2 = feat[:n*num_part//2], feat[n*num_part//2:]
        feat1 = rearrange(feat1, '(n n_h) c s w_h w -> n c s (n_h w_h) w',n_h=num_part)
        feat2 = rearrange(feat2, '(n n_h) c s w_h w -> n c s (n_h w_h) w',n_h=(num_part+1))
        pad_crop = win_size // 8 // (16 // self.cfgs['model_cfg']['bin_num'][0])
        feat2 = feat2[...,pad_crop:-pad_crop,:]
        return torch.concat([feat1, feat2], dim=0) # n c s h w

    def forward(self, inputs):
        if self.training:
            (sils_q, sils_k), labs, typs, vies, (seqL_q, seqL_k) = inputs
            sils_q, sils_k = sils_q[0].unsqueeze(1), sils_k[0].unsqueeze(1)
            n,c,s,h,w = sils_q.shape
            assert h==64 and w==44

            # Teacher Branch
            with torch.no_grad():
                q_input = (sils_q, seqL_q)
                _, _, _, z1, _ = self.teacher_backbone(q_input)

            # Student Branch
            crop_list = self.crop_list
            sils_v_list = torch.chunk(sils_k, len(crop_list), dim=0)
            feat2 = []
            for i, _ in enumerate(sils_v_list):
                tmp = cp.checkpoint(self.diverse_crop_encode, _, crop_list[i], use_reentrant=False)
                feat2.append(tmp)
            feat2 = torch.concat(feat2, dim=0)
            feat2 = self.student_backbone.TP(feat2, seqL_k, options={"dim": 2})[0] # [n, c, h, w]
            feat2 = self.student_backbone.HPP(feat2) # [n, c, p], Horizontal Pooling, HP
            feat2 = self.student_backbone.projector(feat2)
            feat2 = self.student_backbone.predictor(feat2)

            logits1, labels1 = self.D(feat2, z1)
            
            retval = {
                    'training_feat': {
                        'softmax': {'logits': logits1, 'labels': labels1},
                    },
                    'visual_summary': {
                        'image/encoder_q': rearrange(sils_q, 'n c s h w -> (n s) c h w'),
                        'image/encoder_k': rearrange(sils_k, 'n c s h w -> (n s) c h w'),
                    },
                    'inference_feat': None
            }
            return retval
        else:
            sils, labs, typs, vies, seqL = inputs
            sils = sils[0].unsqueeze(1)
            
            assert sils.shape[-1] == 44
            # sils = F.pad(sils, (2, 2))
            # assert sils.shape[-1] == 48

            _, _, _, _, feat = self.teacher_backbone((sils, seqL))
            embed_list = torch.chunk(feat, feat.shape[-1], dim=-1)
            retval = {
                'training_feat': None,
                'visual_summary': None,
                'inference_feat': {
                    'embeddings': F.normalize(feat, dim=1),
                    **{f'embeddings_{i:02d}': embed_list[i] for i in range(feat.shape[-1])}
                    }
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


class FoundationGait_Scaling_Finetune(BaseModel):
    def build_network(self, model_cfg):
        self.teacher_backbone = DeepGaitV2_SSL(model_cfg)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.projector_lr = model_cfg['projector_lr']
        self.backbone_lr = model_cfg['backbone_lr']
        self.crop_list = model_cfg['crop_list']

    def diverse_crop_encode(self, x, num_part):
        n,c,s,h,w = x.shape
        assert n % 2 == 0
        if num_part == 1:
            return self.teacher_backbone.encoder(x)[-1]
        win_size = h//num_part
        x1, x2 = x[:n//2], x[n//2:]
        x1 = rearrange(x1, 'n c s (n_h w_h) w -> (n n_h) c s w_h w',n_h=num_part)
        x2 = F.pad(x2, (0, 0, win_size//2, win_size//2))
        x2 = x2.unfold(dimension=-2, size=win_size, step=win_size)
        x2 = rearrange(x2, 'n c s n_h w w_h -> (n n_h) c s w_h w',n_h=(num_part+1))
        feat = self.teacher_backbone.encoder(torch.concat([x1,x2]))[-1]
        feat1, feat2 = feat[:n*num_part//2], feat[n*num_part//2:]
        feat1 = rearrange(feat1, '(n n_h) c s w_h w -> n c s (n_h w_h) w',n_h=num_part)
        feat2 = rearrange(feat2, '(n n_h) c s w_h w -> n c s (n_h w_h) w',n_h=(num_part+1))
        pad_crop = win_size // 8 // (16 // self.cfgs['model_cfg']['bin_num'][0])
        feat2 = feat2[...,pad_crop:-pad_crop,:]
        return torch.concat([feat1, feat2], dim=0) # n c s h w

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = getattr(optim, optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])

        ft_param_list  = []
        self.fix_layer = []
        for i, ft_lr in enumerate(self.backbone_lr):
            if ft_lr != 0:
                ft_param_list.append({
                    'params': getattr(self.teacher_backbone, 'layer%d'%(i)).parameters(), 
                    'lr': ft_lr, 
                })
            else:
                self.fix_layer.append('layer%d'%(i))

        ft_param_list.append({
            'params': self.teacher_backbone.projector.parameters(), 
            'lr': self.projector_lr, 
        })
        ft_param_list.append({
            'params': self.teacher_backbone.predictor.parameters(), 
            'lr': self.projector_lr, 
        })
        ft_param_list.append({
            'params': self.FCs.parameters(), 
            'lr': valid_arg['lr']
        })
        ft_param_list.append({
            'params': self.BNNecks.parameters(), 
            'lr': valid_arg['lr']
        })

        optimizer = optimizer(ft_param_list, **valid_arg)

        return optimizer

    def forward(self, inputs):
        # if self.training:
        #     self.maintain_non_zero_learning_rate()

        sils, labs, typs, vies, seqL = inputs
        sils = sils[0].unsqueeze(1)
        
        # Version 3 Finetuning with Crop DA
        crop_list = self.crop_list
        sils_v_list = torch.chunk(sils, len(crop_list), dim=0)
        feat = []
        for i, _ in enumerate(sils_v_list):
            if self.training:
                tmp = cp.checkpoint(self.diverse_crop_encode, _, crop_list[i], use_reentrant=False)
            else:
                tmp = cp.checkpoint(self.teacher_backbone.encoder, _, use_reentrant=False)[-1]
            feat.append(tmp)
        feat = torch.concat(feat, dim=0)
        feat = self.teacher_backbone.TP(feat, seqL, options={"dim": 2})[0] # [n, c, h, w]
        feat = self.teacher_backbone.HPP(feat) # [n, c, p], Horizontal Pooling, HP
        feat = self.teacher_backbone.projector(feat)
        feat = F.normalize(feat, dim=1)
        embed_1 = self.FCs(feat) # n,c,p
        _, logits = self.BNNecks(embed_1) # n,class num,p

        embed_list = torch.chunk(embed_1, embed_1.shape[-1], dim=-1)
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed_1,
                **{f'embeddings_{i:02d}': embed_list[i] for i in range(embed_1.shape[-1])}
            }
        }
        return retval
    
    def maintain_non_zero_learning_rate(self):
        if self.iteration % 1000 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] < 1e-4:
                    param_group['lr'] = 1e-4
                    

class FoundationGait_Scaling_LinearProbing_Scoliosis1K(BaseModel):
    def build_network(self, model_cfg):
        self.teacher_backbone = DeepGaitV2_SSL(model_cfg)
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.projector_lr = model_cfg['projector_lr']
        self.backbone_lr = model_cfg['backbone_lr']
        self.crop_list = model_cfg['crop_list']

    def diverse_crop_encode(self, x, num_part):
        n,c,s,h,w = x.shape
        assert n % 2 == 0
        if num_part == 1:
            return self.teacher_backbone.encoder(x)[-1]
        win_size = h//num_part
        x1, x2 = x[:n//2], x[n//2:]
        x1 = rearrange(x1, 'n c s (n_h w_h) w -> (n n_h) c s w_h w',n_h=num_part)
        x2 = F.pad(x2, (0, 0, win_size//2, win_size//2))
        x2 = x2.unfold(dimension=-2, size=win_size, step=win_size)
        x2 = rearrange(x2, 'n c s n_h w w_h -> (n n_h) c s w_h w',n_h=(num_part+1))
        feat = self.teacher_backbone.encoder(torch.concat([x1,x2]))[-1]
        feat1, feat2 = feat[:n*num_part//2], feat[n*num_part//2:]
        feat1 = rearrange(feat1, '(n n_h) c s w_h w -> n c s (n_h w_h) w',n_h=num_part)
        feat2 = rearrange(feat2, '(n n_h) c s w_h w -> n c s (n_h w_h) w',n_h=(num_part+1))
        pad_crop = win_size // 8 // (16 // self.cfgs['model_cfg']['bin_num'][0])
        feat2 = feat2[...,pad_crop:-pad_crop,:]
        return torch.concat([feat1, feat2], dim=0) # n c s h w

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = getattr(optim, optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])

        ft_param_list  = []
        self.fix_layer = []
        for i, ft_lr in enumerate(self.backbone_lr):
            if ft_lr != 0:
                ft_param_list.append({
                    'params': getattr(self.teacher_backbone, 'layer%d'%(i)).parameters(), 
                    'lr': ft_lr, 
                })
            else:
                self.fix_layer.append('layer%d'%(i))

        ft_param_list.append({
            'params': self.teacher_backbone.projector.parameters(), 
            'lr': self.projector_lr, 
        })
        ft_param_list.append({
            'params': self.teacher_backbone.predictor.parameters(), 
            'lr': self.projector_lr, 
        })
        ft_param_list.append({
            'params': self.BNNecks.parameters(), 
            'lr': valid_arg['lr']
        })

        optimizer = optimizer(ft_param_list, **valid_arg)

        return optimizer

    def forward(self, inputs):
        # if self.training:
        #     self.maintain_non_zero_learning_rate()

        sils, labs, typs, vies, seqL = inputs
        sils = sils[0].unsqueeze(1)
        
        label_ids = np.array([{'negative': 0, 'neutral': 1, 'positive': 2}[status] for status in typs])
        label_ids = torch.from_numpy(label_ids).to(sils.device).long()

        crop_list = self.crop_list
        sils_v_list = torch.chunk(sils, len(crop_list), dim=0)
        feat = []
        for i, _ in enumerate(sils_v_list):
            if self.training:
                tmp = cp.checkpoint(self.diverse_crop_encode, _, crop_list[i], use_reentrant=False)
            else:
                tmp = cp.checkpoint(self.teacher_backbone.encoder, _, use_reentrant=False)[-1]
            feat.append(tmp)
        feat = torch.concat(feat, dim=0)
        feat = self.teacher_backbone.TP(feat, seqL, options={"dim": 2})[0] # [n, c, h, w]
        feat = self.teacher_backbone.HPP(feat) # [n, c, p], Horizontal Pooling, HP
        feat = self.teacher_backbone.projector(feat)
        feat = F.normalize(feat, dim=1)
        _, logits = self.BNNecks(rearrange(feat, 'n c (p x) -> n (c p) x', x=1).contiguous())
        logits_list = torch.chunk(logits, logits.shape[-1], dim=-1)
        retval = {
            'training_feat': {
                'softmax': {'logits': logits, 'labels': label_ids},
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': logits,
                **{f'embeddings_{i:02d}': logits_list[i] for i in range(logits.shape[-1])}
            }
        }
        return retval
    
    def maintain_non_zero_learning_rate(self):
        if self.iteration % 1000 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] < 1e-4:
                    param_group['lr'] = 1e-4
                    

class FoundationGait_Scaling_LinearProbing_DGait(BaseModel):
    def build_network(self, model_cfg):
        self.teacher_backbone = DeepGaitV2_SSL(model_cfg)
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.projector_lr = model_cfg['projector_lr']
        self.backbone_lr = model_cfg['backbone_lr']
        self.crop_list = model_cfg['crop_list']

    def diverse_crop_encode(self, x, num_part):
        n,c,s,h,w = x.shape
        assert n % 2 == 0
        if num_part == 1:
            return self.teacher_backbone.encoder(x)[-1]
        win_size = h//num_part
        x1, x2 = x[:n//2], x[n//2:]
        x1 = rearrange(x1, 'n c s (n_h w_h) w -> (n n_h) c s w_h w',n_h=num_part)
        x2 = F.pad(x2, (0, 0, win_size//2, win_size//2))
        x2 = x2.unfold(dimension=-2, size=win_size, step=win_size)
        x2 = rearrange(x2, 'n c s n_h w w_h -> (n n_h) c s w_h w',n_h=(num_part+1))
        feat = self.teacher_backbone.encoder(torch.concat([x1,x2]))[-1]
        feat1, feat2 = feat[:n*num_part//2], feat[n*num_part//2:]
        feat1 = rearrange(feat1, '(n n_h) c s w_h w -> n c s (n_h w_h) w',n_h=num_part)
        feat2 = rearrange(feat2, '(n n_h) c s w_h w -> n c s (n_h w_h) w',n_h=(num_part+1))
        pad_crop = win_size // 8 // (16 // self.cfgs['model_cfg']['bin_num'][0])
        feat2 = feat2[...,pad_crop:-pad_crop,:]
        return torch.concat([feat1, feat2], dim=0) # n c s h w

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = getattr(optim, optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])

        ft_param_list  = []
        self.fix_layer = []
        for i, ft_lr in enumerate(self.backbone_lr):
            if ft_lr != 0:
                ft_param_list.append({
                    'params': getattr(self.teacher_backbone, 'layer%d'%(i)).parameters(), 
                    'lr': ft_lr, 
                })
            else:
                self.fix_layer.append('layer%d'%(i))

        ft_param_list.append({
            'params': self.teacher_backbone.projector.parameters(), 
            'lr': self.projector_lr, 
        })
        ft_param_list.append({
            'params': self.teacher_backbone.predictor.parameters(), 
            'lr': self.projector_lr, 
        })
        ft_param_list.append({
            'params': self.BNNecks.parameters(), 
            'lr': valid_arg['lr']
        })

        optimizer = optimizer(ft_param_list, **valid_arg)

        return optimizer

    def forward(self, inputs):
        # if self.training:
        #     self.maintain_non_zero_learning_rate()

        sils, labs, typs, vies, seqL = inputs
        sils = sils[0].unsqueeze(1)
        
        label_ids = np.array([1 if 'D' in status else 0 for status in typs])
        label_ids = torch.from_numpy(label_ids).to(sils.device).long()
        
        crop_list = self.crop_list
        sils_v_list = torch.chunk(sils, len(crop_list), dim=0)
        feat = []
        for i, _ in enumerate(sils_v_list):
            if self.training:
                tmp = cp.checkpoint(self.diverse_crop_encode, _, crop_list[i], use_reentrant=False)
            else:
                tmp = cp.checkpoint(self.teacher_backbone.encoder, _, use_reentrant=False)[-1]
            feat.append(tmp)
        feat = torch.concat(feat, dim=0)
        feat = self.teacher_backbone.TP(feat, seqL, options={"dim": 2})[0] # [n, c, h, w]
        feat = self.teacher_backbone.HPP(feat) # [n, c, p], Horizontal Pooling, HP
        feat = self.teacher_backbone.projector(feat)
        feat = F.normalize(feat, dim=1)
        _, logits = self.BNNecks(rearrange(feat, 'n c (p x) -> n (c p) x', x=1).contiguous())
        logits_list = torch.chunk(logits, logits.shape[-1], dim=-1)
        retval = {
            'training_feat': {
                'softmax': {'logits': logits, 'labels': label_ids},
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': logits,
                **{f'embeddings_{i:02d}': logits_list[i] for i in range(logits.shape[-1])}
            }
        }
        return retval
    
    def maintain_non_zero_learning_rate(self):
        if self.iteration % 1000 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] < 1e-4:
                    param_group['lr'] = 1e-4


class FoundationGait_Scaling_LinearProbing_RAGAR(BaseModel):
    def build_network(self, model_cfg):
        self.teacher_backbone = DeepGaitV2_SSL(model_cfg)
        self.Attribute_list = model_cfg['Attribute_list']
        self.BNNeck_list = nn.ModuleList([
            SeparateBNNecks(class_num=2, in_channels=model_cfg['SeparateBNNecks']['in_channels'], parts_num=1) for _ in range(sum(self.Attribute_list))
        ])
        self.projector_lr = model_cfg['projector_lr']
        self.backbone_lr = model_cfg['backbone_lr']
        self.crop_list = model_cfg['crop_list']

    def diverse_crop_encode(self, x, num_part):
        n,c,s,h,w = x.shape
        assert n % 2 == 0
        if num_part == 1:
            return self.teacher_backbone.encoder(x)[-1]
        win_size = h//num_part
        x1, x2 = x[:n//2], x[n//2:]
        x1 = rearrange(x1, 'n c s (n_h w_h) w -> (n n_h) c s w_h w',n_h=num_part)
        x2 = F.pad(x2, (0, 0, win_size//2, win_size//2))
        x2 = x2.unfold(dimension=-2, size=win_size, step=win_size)
        x2 = rearrange(x2, 'n c s n_h w w_h -> (n n_h) c s w_h w',n_h=(num_part+1))
        feat = self.teacher_backbone.encoder(torch.concat([x1,x2]))[-1]
        feat1, feat2 = feat[:n*num_part//2], feat[n*num_part//2:]
        feat1 = rearrange(feat1, '(n n_h) c s w_h w -> n c s (n_h w_h) w',n_h=num_part)
        feat2 = rearrange(feat2, '(n n_h) c s w_h w -> n c s (n_h w_h) w',n_h=(num_part+1))
        pad_crop = win_size // 8 // (16 // self.cfgs['model_cfg']['bin_num'][0])
        feat2 = feat2[...,pad_crop:-pad_crop,:]
        return torch.concat([feat1, feat2], dim=0) # n c s h w

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = getattr(optim, optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])

        ft_param_list  = []
        self.fix_layer = []
        for i, ft_lr in enumerate(self.backbone_lr):
            if ft_lr != 0:
                ft_param_list.append({
                    'params': getattr(self.teacher_backbone, 'layer%d'%(i)).parameters(), 
                    'lr': ft_lr, 
                })
            else:
                self.fix_layer.append('layer%d'%(i))

        ft_param_list.append({
            'params': self.teacher_backbone.projector.parameters(), 
            'lr': self.projector_lr, 
        })
        ft_param_list.append({
            'params': self.teacher_backbone.predictor.parameters(), 
            'lr': self.projector_lr, 
        })
        for i,j in enumerate(self.BNNeck_list):
            ft_param_list.append({
                'params': self.BNNeck_list[i].parameters(), 
                'lr': valid_arg['lr']
            })

        optimizer = optimizer(ft_param_list, **valid_arg)

        return optimizer

    def forward(self, inputs):
        # if self.training:
        #     self.maintain_non_zero_learning_rate()

        sils, labs, typs, vies, seqL = inputs
        sils = sils[0].unsqueeze(1)
        
        label_ids = np.array([np.array(status.split('_'), dtype=int)[1:] for status in typs])
        binary_labels = []
        for i, n_cls in enumerate(self.Attribute_list):
            for c in range(n_cls):
                binary_labels.append((label_ids[:, i] == c).astype(int))
        binary_labels = np.stack(binary_labels, axis=1)
        label_ids = torch.from_numpy(binary_labels).to(sils.device).long()
        
        crop_list = self.crop_list
        sils_v_list = torch.chunk(sils, len(crop_list), dim=0)
        feat = []
        for i, _ in enumerate(sils_v_list):
            if self.training:
                tmp = cp.checkpoint(self.diverse_crop_encode, _, crop_list[i], use_reentrant=False)
            else:
                tmp = cp.checkpoint(self.teacher_backbone.encoder, _, use_reentrant=False)[-1]
            feat.append(tmp)
        feat = torch.concat(feat, dim=0)
        feat = self.teacher_backbone.TP(feat, seqL, options={"dim": 2})[0] # [n, c, h, w]
        feat = self.teacher_backbone.HPP(feat) # [n, c, p], Horizontal Pooling, HP
        feat = self.teacher_backbone.projector(feat)
        feat = F.normalize(feat, dim=1)
        # feat = self.FCs(rearrange(feat, 'n c (p x) -> n (c p) x', x=1).contiguous()) # [n cp 1]
        feat = rearrange(feat, 'n c (p x) -> n (c p) x', x=1).contiguous()
        logits_list = []
        for bnneck in self.BNNeck_list:
            _, logits = bnneck(feat)
            logits_list.append(logits)
        retval = {
            'training_feat': {
                **{f'softmax_{i:02d}': {'logits': logits_list[i], 'labels': label_ids[:,i]} for i in range(len(logits_list))}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': torch.concat(logits_list, dim=-1)
            }
        }
        return retval
    
    def maintain_non_zero_learning_rate(self):
        if self.iteration % 1000 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] < 1e-4:
                    param_group['lr'] = 1e-4
                    
