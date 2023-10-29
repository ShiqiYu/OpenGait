import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from torch.nn import functional as F
import numpy as np
from ..backbones.gcn import GCN


def L_Matrix(adj_npy, adj_size):

    D =np.zeros((adj_size, adj_size))
    for i in range(adj_size):
        tmp = adj_npy[i,:]
        count = np.sum(tmp==1)
        if count>0:
            number = count ** (-1/2)
            D[i,i] = number

    x = np.matmul(D,adj_npy)
    L = np.matmul(x,D)
    return L

def get_fine_adj_npy():
    fine_adj_list = [
        # 1  2  3  4  5  6  7  8  9  10 11
        [ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], #1
        [ 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], #2
        [ 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1], #3
        [ 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1], #4
        [ 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], #5
        [ 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], #6
        [ 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1], #7
        [ 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1], #8
        [ 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], #9
        [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], #10
        [ 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]  #11
    ]
    fine_adj_npy = np.array(fine_adj_list)
    fine_adj_npy = L_Matrix(fine_adj_npy, len(fine_adj_npy))   # len返回的是行数
    return fine_adj_npy

def get_coarse_adj_npy():
    coarse_adj_list = [
        # 1  2  3  4  5
        [ 1, 1, 1, 1, 1], #1
        [ 1, 1, 0, 0, 0], #2
        [ 1, 0, 1, 0, 0], #3
        [ 1, 0, 0, 1, 0], #4
        [ 1, 0, 0, 0, 1]  #5
    ]
    coarse_adj_npy = np.array(coarse_adj_list)
    coarse_adj_npy = L_Matrix(coarse_adj_npy, len(coarse_adj_npy))   # len返回的是行数
    return coarse_adj_npy


class ParsingGait(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        nfeat = model_cfg['SeparateFCs']['in_channels']
        gcn_cfg = model_cfg['gcn_cfg']
        self.fine_parts = gcn_cfg['fine_parts']
        coarse_parts = gcn_cfg['coarse_parts']

        self.only_fine_graph = gcn_cfg['only_fine_graph']
        self.only_coarse_graph = gcn_cfg['only_coarse_graph']
        self.combine_fine_coarse_graph = gcn_cfg['combine_fine_coarse_graph']

        if self.only_fine_graph:
            fine_adj_npy = get_fine_adj_npy()
            self.fine_adj_npy = torch.from_numpy(fine_adj_npy).float()
            self.gcn_fine = GCN(self.fine_parts, nfeat, nfeat, isMeanPooling=True)
            self.gammas_fine = torch.nn.Parameter(torch.ones(self.fine_parts) * 0.75)
        elif self.only_coarse_graph:
            coarse_adj_npy = get_coarse_adj_npy()
            self.coarse_adj_npy = torch.from_numpy(coarse_adj_npy).float()
            self.gcn_coarse = GCN(coarse_parts, nfeat, nfeat, isMeanPooling=True)
            self.gammas_coarse = torch.nn.Parameter(torch.ones(coarse_parts) * 0.75)
        elif self.combine_fine_coarse_graph:
            fine_adj_npy = get_fine_adj_npy()
            self.fine_adj_npy = torch.from_numpy(fine_adj_npy).float()
            self.gcn_fine = GCN(self.fine_parts, nfeat, nfeat, isMeanPooling=True)
            self.gammas_fine = torch.nn.Parameter(torch.ones(self.fine_parts) * 0.75)
            coarse_adj_npy = get_coarse_adj_npy()
            self.coarse_adj_npy = torch.from_numpy(coarse_adj_npy).float()
            self.gcn_coarse = GCN(coarse_parts, nfeat, nfeat, isMeanPooling=True)
            self.gammas_coarse = torch.nn.Parameter(torch.ones(coarse_parts) * 0.75)
        else:
            raise ValueError("You should choose fine/coarse graph, or combine both of them.")
        
    def PPforGCN(self, x):
        """
            Part Pooling for GCN
            x   : [n, p, c, h, w]
            ret : [n, p, c] 
        """
        n, p, c, h, w = x.size()
        z = x.view(n, p, c, -1)  # [n, p, c, h*w]
        z = z.mean(-1) + z.max(-1)[0]   # [n, p, c]
        return z
    
    def ParsPartforFineGraph(self, mask_resize, z):
        """
            x: [n, c, s, h, w]
            paes: [n, 1, s, H, W]
            return [n*s, 11, c, h, w]
            ***Fine Parts:
            # 0: Background, 
            1: Head, 
            2: Torso, 
            3: Left-arm, 
            4: Right-arm, 
            5: Left-hand, 
            6: Right-hand, 
            7: Left-leg, 
            8: Right-leg, 
            9: Left-foot, 
            10: Right-foot, 
            11: Dress
        """
        fine_mask_list = list()
        for i in range(1, self.fine_parts + 1):
            fine_mask_list.append((mask_resize.long() == i)) # split mask of each class

        fine_z_list = list()
        for i in range(len(fine_mask_list)):
            mask = fine_mask_list[i].unsqueeze(1)
            fine_z_list.append((mask.float() * z * self.gammas_fine[i] + (~mask).float() * z * (1.0 - self.gammas_fine[i])).unsqueeze(1)) # split feature map by mask of each class
        fine_z_feat = torch.cat(fine_z_list, dim=1)  # [n*s, 11, c, h, w] or [n*s, 5, c, h, w]
        
        return fine_z_feat

    def ParsPartforCoarseGraph(self, mask_resize, z):
        """
            x: [n, c, s, h, w]
            paes: [n, 1, s, H, W]
            return [n*s, 5, c, h, w]
            ***Coarse Parts:
            1: [1, 2, 11]  Head, Torso, Dress
            2: [3, 5]  Left-arm, Left-hand
            3: [4, 6]  Right-arm, Right-hand
            4: [7, 9]  Left-leg, Left-foot
            5: [8, 10] Right-leg, Right-foot
        """
        coarse_mask_list = list()
        coarse_parts = [[1,2,11], [3,5], [4,6], [7,9], [8,10]]
        for coarse_part in coarse_parts:
            part = mask_resize.long() == -1
            for i in coarse_part:
                part += (mask_resize.long() == i)
            coarse_mask_list.append(part)

        coarse_z_list = list()
        for i in range(len(coarse_mask_list)):
            mask = coarse_mask_list[i].unsqueeze(1)
            coarse_z_list.append((mask.float() * z * self.gammas_coarse[i] + (~mask).float() * z * (1.0 - self.gammas_coarse[i])).unsqueeze(1)) # split feature map by mask of each class
        coarse_z_feat = torch.cat(coarse_z_list, dim=1)  # [n*s, 11, c, h, w] or [n*s, 5, c, h, w]

        return coarse_z_feat

    def ParsPartforGCN(self, x, pars):
        """
            x: [n, c, s, h, w]
            paes: [n, 1, s, H, W]
            return [n*s, 11, c, h, w] or [n*s, 5, c, h, w]
        """
        n, c, s, h, w = x.size()
        # mask_resize: [n, s, h, w]
        mask_resize = F.interpolate(input=pars.squeeze(1), size=(h, w), mode='nearest')
        mask_resize = mask_resize.view(n*s, h, w)

        z = x.transpose(1, 2).reshape(n*s, c, h, w)
        
        if self.only_fine_graph:
            fine_z_feat = self.ParsPartforFineGraph(mask_resize, z)
            return fine_z_feat, None
        elif self.only_coarse_graph:
            coarse_z_feat = self.ParsPartforCoarseGraph(mask_resize, z)
            return None, coarse_z_feat
        elif self.combine_fine_coarse_graph:
            fine_z_feat = self.ParsPartforFineGraph(mask_resize, z)
            coarse_z_feat = self.ParsPartforCoarseGraph(mask_resize, z)
            return fine_z_feat, coarse_z_feat
        else:
            raise ValueError("You should choose fine/coarse graph, or combine both of them.")


    def get_gcn_feat(self, n, input, adj_np, is_cuda, seqL):
        input_ps = self.PPforGCN(input)  # [n*s, 11, c]
        n_s, p, c = input_ps.size()
        if is_cuda:
            adj = adj_np.cuda()
        adj = adj.repeat(n_s, 1, 1)
        if p == 11:
            output_ps = self.gcn_fine(input_ps, adj)  # [n*s, 11, c]
        elif p == 5:
            output_ps = self.gcn_coarse(input_ps, adj)  # [n*s, 5, c]
        else:
            raise ValueError(f"The parsing parts should be 11 or 5, but got {p}")
        output_ps = output_ps.view(n, n_s//n, p, c)   # [n, s, ps, c]
        output_ps = self.TP(output_ps, seqL, dim=1, options={"dim": 1})[0]  # [n, ps, c]

        return output_ps


    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        pars = ipts[0]
        if len(pars.size()) == 4:
            pars = pars.unsqueeze(1)

        del ipts
        outs = self.Backbone(pars)  # [n, c, s, h, w]

        outs_n, outs_c, outs_s, outs_h, outs_w = outs.size()

        # split features by parsing classes
        # outs_ps_fine: [n*s, 11, c, h, w]
        # outs_ps_coarse: [n*s, 5, c, h, w]
        outs_ps_fine, outs_ps_coarse = self.ParsPartforGCN(outs, pars)

        is_cuda = pars.is_cuda
        if self.only_fine_graph:
            outs_ps = self.get_gcn_feat(outs_n, outs_ps_fine, self.fine_adj_npy, is_cuda, seqL)  # [n, 11, c]
        elif self.only_coarse_graph:
            outs_ps = self.get_gcn_feat(outs_n, outs_ps_coarse, self.coarse_adj_npy, is_cuda, seqL)  # [n, 5, c]
        elif self.combine_fine_coarse_graph:
            outs_fine = self.get_gcn_feat(outs_n, outs_ps_fine, self.fine_adj_npy, is_cuda, seqL)  # [n, 11, c]
            outs_coarse = self.get_gcn_feat(outs_n, outs_ps_coarse, self.coarse_adj_npy, is_cuda, seqL)  # [n, 5, c]
            outs_ps = torch.cat([outs_fine, outs_coarse], 1)  # [n, 16, c]
        else:
            raise ValueError("You should choose fine/coarse graph, or combine both of them.")
        outs_ps = outs_ps.transpose(1, 2).contiguous()  # [n, c, ps]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        feat = torch.cat([feat, outs_ps], dim=-1)  # [n, c, p+ps]

        embed_1 = self.FCs(feat)  # [n, c, p+ps]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p+ps]
        embed = embed_1

        n, _, s, h, w = pars.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/pars': pars.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
