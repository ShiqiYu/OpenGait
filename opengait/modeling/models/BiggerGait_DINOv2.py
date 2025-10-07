# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from kornia import morphology as morph
import random

from .BigGait_utils.BigGait_GaitBase import *
from .BigGait_utils.save_img import save_image, pca_image
from functools import partial

# ######################################## BiggerGait ###########################################

class infoDistillation(nn.Module):
    def __init__(self, source_dim, target_dim, p, softmax):
        super(infoDistillation, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        self.down_sampling = nn.Linear(source_dim, target_dim)
        self.up_sampling = nn.Linear(target_dim, source_dim)
        self.softmax = softmax
        self.mse = nn.MSELoss()

    def forward(self, x, mse=True):
        # [n, c]
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        d_x = F.softmax(d_x, dim=1)
        if mse:
            u_x = self.up_sampling(d_x)
            return d_x, torch.mean(self.mse(u_x, x))
        else:
            return d_x, None

class ResizeToHW(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

class BiggerGait__DINOv2(BaseModel):
    def build_network(self, model_cfg):
        # get pretained models
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]

        # set input size
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]

        # set feature dim
        self.f4_dim = model_cfg['source_dim']
        self.num_unknown = model_cfg["num_unknown"]

        # set layer / group / gait_head number
        self.total_layer_num = model_cfg["total_layer_num"] # total layer number is 12
        self.group_layer_num = model_cfg["group_layer_num"] # each group have 2 layers
        self.head_num = model_cfg["head_num"] # 2 gait heads
        assert self.total_layer_num % self.group_layer_num == 0
        assert (self.total_layer_num // self.group_layer_num) % self.head_num == 0
        self.num_FPN = self.total_layer_num // self.group_layer_num

        self.Gait_Net = Baseline_Share(model_cfg)

        self.HumanSpace_Conv = nn.ModuleList([ 
                nn.Sequential(
                    nn.BatchNorm2d(self.f4_dim*self.group_layer_num, affine=False),
                    nn.Conv2d(self.f4_dim*self.group_layer_num, self.f4_dim//2, kernel_size=1),
                    nn.BatchNorm2d(self.f4_dim//2, affine=False),
                    nn.GELU(),
                    nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                    ResizeToHW((self.sils_size*2, self.sils_size)),
                    nn.BatchNorm2d(self.num_unknown, affine=False),
                    nn.Sigmoid()
                ) for _ in range(self.num_FPN)
            ])
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])

    def init_DINOv2(self):
        from transformers import Dinov2Config, Dinov2Model
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        config = Dinov2Config.from_pretrained(self.pretrained_lvm + "/config.json")
        self.Backbone = Dinov2Model.from_pretrained(
            self.pretrained_lvm, 
            config=config,
        )
        self.Backbone.cpu()
        self.msg_mgr.log_info(f'load model from: {self.pretrained_lvm}')

    def init_Mask_Branch(self):
        self.msg_mgr.log_info(f'load model from: {self.pretrained_mask_branch}')
        load_dict = torch.load(self.pretrained_mask_branch, map_location=torch.device("cpu"))['model']
        msg = self.Mask_Branch.load_state_dict(load_dict, strict=True)
        n_parameters = sum(p.numel() for p in self.Mask_Branch.parameters())
        self.msg_mgr.log_info('Missing keys: {}'.format(msg.missing_keys))
        self.msg_mgr.log_info('Unexpected keys: {}'.format(msg.unexpected_keys))
        self.msg_mgr.log_info(f"=> loaded successfully '{self.pretrained_mask_branch}'")
        self.msg_mgr.log_info('SegmentationBranch Count: {:.5f}M'.format(n_parameters / 1e6))

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('Expect Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
        
        self.init_DINOv2()
        self.init_Mask_Branch()
        
        # # Cal GFlops
        if self.training:
            from fvcore.nn import FlopCountAnalysis
            self.eval()
            with torch.no_grad():
                device = torch.distributed.get_rank()
                inputs = ([[torch.randn((1,1,3,448,224),dtype=torch.float32).to(device), torch.rand(1,dtype=torch.float32).to(device)], None, None, None, None],)
                flops = FlopCountAnalysis(self.to(device), inputs).total()  / 1e9   # GFLOPs 
            self.train()
        
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.Mask_Branch.eval()
        self.Mask_Branch.requires_grad_(False)
        
        n_parameters = sum(p.numel() for p in self.parameters())
        if self.training:
            self.msg_mgr.log_info('All Backbone Count: {:.5f}M, {:.2f} GFLOPs'.format(n_parameters / 1e6, flops))
        else:
            self.msg_mgr.log_info('All Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
            
        self.msg_mgr.log_info("=> init successfully")

    # resize image
    def preprocess(self, sils, image_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(sils, (image_size*2, image_size), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

# # ############################# For Train ##############################

    def forward(self, inputs):
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        # adjust gpu
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//96)+1, dim=1)
        all_outs = []
        for _, rgb_img in enumerate(rgb_chunks):
            with torch.no_grad():
                # get RGB
                n,s,c,h,w = rgb_img.size()
                rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
                outs = self.preprocess(rgb_img, self.image_size)
                outs = self.Backbone(outs,output_hidden_states=True).hidden_states[1:] # [ns,h*w,c]

                intermediates = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim*len(outs), elementwise_affine=False)(torch.concat(outs, dim=-1))[:,1:]
                intermediates = rearrange(intermediates.view(n, s, self.image_size//7, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
                intermediates = list(torch.chunk(intermediates, self.total_layer_num, dim=1))

                human_mask = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim, elementwise_affine=False)(outs[-1])[:,1:].contiguous()
                human_mask, _ = self.Mask_Branch(human_mask.view(-1, self.f4_dim), mse=False)
                human_mask = (human_mask[:,1] > 0.5).float() # check which is the foreground at first!!!   0 or 1; 50%;
                human_mask = human_mask.view(n*s, 1, self.image_size//7, self.image_size//14)
                human_mask = self.preprocess(human_mask, self.sils_size).detach().clone()

            intermediates = [torch.cat(intermediates[i:i+self.group_layer_num], dim=1).contiguous() for i in range(0, self.total_layer_num, self.group_layer_num)]
            for i in range(self.num_FPN):
                intermediates[i] = self.HumanSpace_Conv[i](intermediates[i])
            intermediates = torch.concat(intermediates, dim=1)
            intermediates = intermediates * (human_mask > 0.5).to(intermediates)
            intermediates = rearrange(intermediates.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> n c s h w').contiguous()

            outs = self.Gait_Net.test_1(intermediates)
            all_outs.append(outs)

        embed_list, log_list = self.Gait_Net.test_2(
                                        torch.cat(all_outs, dim=2),
                                        seqL,
                                        )
        
        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.concat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n*s, c, h, w)[:5].float(),
                    'image/human_mask': self.min_max_norm(human_mask.view(n*s, -1, self.sils_size*2, self.sils_size)[:5].float()),
                },
                'inference_feat': {
                    'embeddings': torch.concat(embed_list, dim=-1),
                }
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},

                'inference_feat': {
                    'embeddings': torch.concat(embed_list, dim=-1),
                }
            }
        return retval
