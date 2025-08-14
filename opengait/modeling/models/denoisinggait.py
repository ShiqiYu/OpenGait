import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, FlowFunc
import torch.optim as optim
from einops import rearrange
from utils import get_valid_args
import warnings
import random
from torchvision.utils import flow_to_image
from .diffgait_utils.GaitBase_fusion_denoise_flow26_attn import GaitBaseFusion_denoise
warnings.filterwarnings("ignore", category=FutureWarning)
from kornia import morphology as morph

import torch.nn as nn
from torch.nn import functional as F
class DenoisingGait(BaseModel):
    def build_network(self, model_cfg):
        self.Backbone = GaitBaseFusion_denoise(model_cfg)

        self.r = model_cfg['r']
        self.p = model_cfg['p']
        self.threshold = model_cfg['threshold']

        self.AppF = AppearanceFunc()
        self.flow3 = FlowFunc(radius=self.r)

        self.AF  = AppFunc_Self()
        self.AF2  = AppFunc_Self()
        self.AF3 = AppFunc_Self()
        self.flow_self = FlowFunc(radius=1)
        self.flow_self2 = FlowFunc(radius=2)
        self.flow_self3 = FlowFunc(radius=3)



    def get_optimizer(self, optimizer_cfg):
        optimizer = getattr(optim, optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])

        ft_param_list  = []
        self.fix_layer = []
        self.ft_param_list_name=[]
        for name, para in self.named_parameters():
            if 'TADPGait' not in name:
                ft_param_list.append(para)
                self.ft_param_list_name.append(name)
                para.requires_grad = True
            else:
                self.fix_layer.append(name)
                para.requires_grad = False

        optimizer = optimizer(ft_param_list, **valid_arg)
        return optimizer
    
    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())
    
    def get_edge(self, sils, threshold=1):
        mask_sils = torch.round(sils * threshold)
        n, c, s, h, w = mask_sils.shape
        mask_sils = rearrange(mask_sils, 'n c s h w -> (n s) c h w')
        kernel = torch.ones((5,5))
        eroded_mask = morph.erosion(mask_sils, kernel.to(sils.device)).detach()  # Erosion
        eroded_sil = (eroded_mask > 0.5) * torch.ones_like(eroded_mask, dtype=eroded_mask.dtype, device=eroded_mask.device)
        eroded_sil = rearrange(eroded_sil, '(n s) c h w -> n c s h w',n=n)
        return eroded_sil
    
    def suppress_large_vectors(self, feature_self_input1, threshold=1.0):
        n, c, s, h, w = feature_self_input1.size()
        feature_self_input1 = rearrange(feature_self_input1, 'n c s h w -> (n s) c h w')
        ns, c, h, w = feature_self_input1.size()

        # Calculate magnitudes of the vectors
        magnitudes = torch.sqrt(torch.sum(feature_self_input1 ** 2, dim=1, keepdim=True))  # [ns, 1, h, w]

        # Find where the magnitudes exceed the threshold
        mask = magnitudes > threshold  # [ns, 1, h, w]

        # Dilate the mask to affect a 3x3 neighborhood around each point
        kernel = torch.ones((1, 1, 3, 3), device=feature_self_input1.device)
        dilated_mask = F.conv2d(mask.float(), kernel, padding=1, groups=1)
        dilated_mask = dilated_mask > 0  # Convert back to a boolean mask

        # Create a masked version of the original features where dilated magnitudes are set to zero
        suppressed_features = feature_self_input1 * (~dilated_mask).float()

        # Reshape back to original dimensions if necessary
        final_features = rearrange(suppressed_features, '(n s) c h w -> n c s h w', n=n, s=s)

        return final_features
        

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        noises = ipts[0]
        mask = ipts[1]
        n, s, c, h, w = noises.shape

        mask = mask.unsqueeze(1) #  n c s h w
        mask[mask>0]=1.
        noises = rearrange(noises, 'n s c h w -> n c s h w',n=n)

        noises_1 = noises
        noises_2 = noises[:, : ,1:, ...]
        noises_2 = torch.cat((noises_2, noises[:, :, -1, :, :].unsqueeze(2)), dim=2)
        noises_input = noises * mask

        mask1 = mask
        mask2 = mask[:, : ,1:, ...]
        mask2 = torch.cat((mask2, mask[:, :, -1, :, :].unsqueeze(2)), dim=2)


        noises_self0, noises_self1 = self.AF(noises, mask)
        noises_self0_1, noises_self1_1 = self.AF2(noises, mask)
        noises_self0_2, noises_self1_2 = self.AF3(noises, mask)


        feature_self_input1 = self.flow_self(noises_self0,noises_self1)
        feature_self_input2 = self.flow_self2(noises_self0_1,noises_self1_1)
        feature_self_input3 = self.flow_self3(noises_self0_2,noises_self1_2)


        eroded_sil = self.get_edge(mask)
        if self.training: 
            idx = random.sample(list(range(n)), int(round(n*self.p)))
            eroded_sil_tmp = eroded_sil[idx]
            eroded_sil_expanded = eroded_sil_tmp.expand(-1, 2, -1, -1, -1)
            eroded_sil_expanded = eroded_sil_expanded.bool()


            feature_self_input1_tmp = feature_self_input1[idx]
            feature_self_input1_tmp_smooth = self.suppress_large_vectors(feature_self_input1_tmp*eroded_sil_tmp, self.threshold)
            combined_features1 = torch.where(eroded_sil_expanded, feature_self_input1_tmp_smooth, feature_self_input1_tmp)
            feature_self_input1[idx] = combined_features1

            feature_self_input2_tmp = feature_self_input2[idx]
            feature_self_input2_tmp_smooth = self.suppress_large_vectors(feature_self_input2_tmp*eroded_sil_tmp, self.threshold)
            combined_features2 = torch.where(eroded_sil_expanded, feature_self_input2_tmp_smooth, feature_self_input2_tmp)
            feature_self_input2[idx] = combined_features2

            feature_self_input3_tmp = feature_self_input3[idx]
            feature_self_input3_tmp_smooth = self.suppress_large_vectors(feature_self_input3_tmp*eroded_sil_tmp, 0.5)
            combined_features3 = torch.where(eroded_sil_expanded, feature_self_input3_tmp_smooth, feature_self_input3_tmp)
            feature_self_input3[idx] = combined_features3


        feature_self_input = torch.cat([feature_self_input1,feature_self_input2,feature_self_input3],dim=1)
        noises_1, noises_2  = self.AppF(noises_1, noises_2, mask1, mask2)
        feature_flow_input = self.flow3(noises_1,noises_2)

        del ipts
        embed_1, logits = self.Backbone(feature_self_input, feature_flow_input, seqL)
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
            },
            'visual_summary': {

            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
    


       
class AppearanceFunc(nn.Module):
    def __init__(self, in_channel=4, out_channel=4, num_heads=2): 
        super(AppearanceFunc, self).__init__()
        self.q_encoder = SetBlockWrapper(nn.Sequential(
                conv1x1(4, 16), 
                nn.BatchNorm2d(16),
                conv3x3(16, 16), 
                nn.BatchNorm2d(16), 
                nn.ReLU(inplace=True), 
                conv3x3(16, 8), 
                nn.BatchNorm2d(8), 
                nn.ReLU(inplace=True), 
                conv1x1(8, 4), 
                nn.BatchNorm2d(4), 
            )
        )
        self.k_encoder = SetBlockWrapper(nn.Sequential(
                conv1x1(4, 16), 
                nn.BatchNorm2d(16), 
                conv3x3(16, 16), 
                nn.BatchNorm2d(16), 
                nn.ReLU(inplace=True), 
                conv3x3(16, 8), 
                nn.BatchNorm2d(8), 
                nn.ReLU(inplace=True), 
                conv1x1(8, 4), 
                nn.BatchNorm2d(4), 
            )
        )

    def forward(self, feature1, feature2, mask1, mask2):
        '''
            features: [n, c, s, h, w]
        '''
        feature1 = self.q_encoder(feature1)
        feature2 = self.k_encoder(feature2)
        feature1 = torch.sigmoid(feature1) * mask1
        feature2 = torch.sigmoid(feature2) * mask2
        return feature1, feature2



class AppFunc_Self(nn.Module):
    def __init__(self, num_heads=2, out_channel=16): 
        super(AppFunc_Self, self).__init__()
        self.encoder0 = SetBlockWrapper(nn.Sequential(
                conv1x1(4, 64), 
                nn.BatchNorm2d(64), 
                conv3x3(64, 64), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True), 
                conv3x3(64, 32), 
                nn.BatchNorm2d(32), 
                nn.ReLU(inplace=True), 
                conv1x1(32, 16), 
                nn.BatchNorm2d(16), 
            )
        )
        self.encoder1 = SetBlockWrapper(nn.Sequential(
                conv1x1(4, 64), 
                nn.BatchNorm2d(64), 
                conv3x3(64, 64), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True), 
                conv3x3(64, 32), 
                nn.BatchNorm2d(32), 
                nn.ReLU(inplace=True), 
                conv1x1(32, 16), 
                nn.BatchNorm2d(16), 
            )
        )

        self.decoder0 = SetBlockWrapper(nn.Sequential(
                conv3x3(16, 16), 
                nn.BatchNorm2d(16), 
                nn.ReLU(inplace=True),)
        )
        self.decoder1 = SetBlockWrapper(nn.Sequential(
                conv3x3(16, 16), 
                nn.BatchNorm2d(16), 
                nn.ReLU(inplace=True),)
        )


    def forward(self, features, mask):
        '''
            features: [n, c, s, h, w]
        '''
        feature_self0 = self.encoder0(features)
        feature_self1 = self.encoder1(features)

        feature_self0 = torch.sigmoid(feature_self0) * mask
        feature_self1 = torch.sigmoid(feature_self1) * mask

        feature_self0 = self.decoder0(feature_self0)
        feature_self1 = self.decoder1(feature_self1)

        return feature_self0, feature_self1
    


