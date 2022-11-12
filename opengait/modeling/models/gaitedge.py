import torch
from kornia import morphology as morph
import torch.optim as optim

from ..base_model import BaseModel
from .gaitgl import GaitGL
from ..modules import GaitAlign
from torchvision.transforms import Resize
from utils import get_valid_args, get_attr_from, is_list_or_tuple
import os.path as osp


class Segmentation(BaseModel):

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        del seqL
        rgbs = ipts[0]
        sils = ipts[1]
        # del ipts
        n, s, c, h, w = rgbs.size()
        rgbs = rgbs.view(n*s, c, h, w)
        sils = sils.view(n*s, 1, h, w)
        logi = self.Backbone(rgbs)  # [n*s, c, h, w]
        logits = torch.sigmoid(logi)
        pred = (logits > 0.5).float()  # [n*s, c, h, w]

        retval = {
            'training_feat': {
                'bce': {'logits': logits, 'labels': sils}
            },
            'visual_summary': {
                'image/sils': sils, 'image/logits': logits, "image/pred": pred
            },
            'inference_feat': {
                'pred': pred, 'mask': sils
            }
        }
        return retval


class GaitEdge(GaitGL):

    def build_network(self, model_cfg):
        super(GaitEdge, self).build_network(model_cfg["GaitGL"])
        self.Backbone = self.get_backbone(model_cfg['Segmentation'])
        self.align = model_cfg['align']
        self.gait_align = GaitAlign()
        self.resize = Resize((64, 44))
        self.is_edge = model_cfg['edge']
        self.seg_lr = model_cfg['seg_lr']
        self.kernel = torch.ones(
            (model_cfg['kernel_size'], model_cfg['kernel_size']))

    def finetune_parameters(self):
        fine_tune_params = list()
        others_params = list()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'Backbone' in name:
                fine_tune_params.append(p)
            else:
                others_params.append(p)
        return [{'params': fine_tune_params, 'lr': self.seg_lr}, {'params': others_params}]

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(self.finetune_parameters(), **valid_arg)
        return optimizer

    def resume_ckpt(self, restore_hint):
        if is_list_or_tuple(restore_hint):
            for restore_hint_i in restore_hint:
                self.resume_ckpt(restore_hint_i)
            return
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def preprocess(self, sils):
        dilated_mask = (morph.dilation(sils, self.kernel.to(sils.device)).detach()
                        ) > 0.5  # Dilation
        eroded_mask = (morph.erosion(sils, self.kernel.to(sils.device)).detach()
                       ) > 0.5   # Erosion
        edge_mask = dilated_mask ^ eroded_mask
        return edge_mask, eroded_mask

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        ratios = ipts[0]
        rgbs = ipts[1]
        sils = ipts[2]

        n, s, c, h, w = rgbs.size()
        rgbs = rgbs.view(n*s, c, h, w)
        sils = sils.view(n*s, 1, h, w)
        logis = self.Backbone(rgbs)  # [n, s, c, h, w]
        logits = torch.sigmoid(logis)
        mask = torch.round(logits).float()
        if self.is_edge:
            edge_mask, eroded_mask = self.preprocess(sils)

            # Gait Synthesis
            new_logits = edge_mask*logits+eroded_mask*sils

            if self.align:
                cropped_logits = self.gait_align(
                    new_logits, sils, ratios)
            else:
                cropped_logits = self.resize(new_logits)
        else:
            if self.align:
                cropped_logits = self.gait_align(
                    logits, mask, ratios)
            else:
                cropped_logits = self.resize(logits)
        _, c, H, W = cropped_logits.size()
        cropped_logits = cropped_logits.view(n, s, H, W)
        retval = super(GaitEdge, self).forward(
            [[cropped_logits], labs, None, None, seqL])
        retval['training_feat']['bce'] = {'logits': logits, 'labels': sils}
        retval['visual_summary']['image/roi'] = cropped_logits.view(
            n*s, 1, H, W)

        return retval
