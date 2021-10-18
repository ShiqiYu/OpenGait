import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks


class Baseline(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg)
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2)

        del ipts
        outs = self.Backbone(sils)  # [n, s, c, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, dim=1)[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]
        feat = feat.permute(2, 0, 1).contiguous()  # [p, n, c]

        embed_1 = self.FCs(feat)  # [p, n, c]
        embed_2, logits = self.BNNecks(embed_1)  # [p, n, c]

        embed_1 = embed_1.permute(1, 0, 2).contiguous()  # [n, p, c]
        embed_2 = embed_2.permute(1, 0, 2).contiguous()  # [n, p, c]
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p, c]
        embed = embed_1

        n, s, _, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
