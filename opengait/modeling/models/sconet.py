import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange
import numpy as np
class ScoNet(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts, pids, labels, _, seqL = inputs

        # Label mapping: negative->0, neutral->1, positive->2 
        label_ids = np.array([{'negative': 0, 'neutral': 1, 'positive': 2}[status] for status in labels])
        label_ids = torch.from_numpy(label_ids).cuda().long()
        
        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': pids},
                'softmax': {'logits': logits, 'labels': label_ids},
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': logits
            }
        }
        return retval