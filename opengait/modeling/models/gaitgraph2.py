import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..backbones.resgcn import ResGCN
from ..modules import Graph
import numpy as np


class GaitGraph2(BaseModel):
    """
        GaitGraph2: Towards a Deeper Understanding of Skeleton-based Gait Recognition
        Paper:    https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper
        Github:   https://github.com/tteepe/GaitGraph2
    """
    def build_network(self, model_cfg):
         
        self.joint_format = model_cfg['joint_format']
        self.input_num = model_cfg['input_num']
        self.block = model_cfg['block']
        self.input_branch = model_cfg['input_branch']
        self.main_stream = model_cfg['main_stream']
        self.num_class = model_cfg['num_class']
        self.reduction = model_cfg['reduction']
        self.tta = model_cfg['tta']
        ## Graph Init ##
        self.graph = Graph(joint_format=self.joint_format,max_hop=3)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        ## Network ##
        self.ResGCN = ResGCN(input_num=self.input_num, input_branch=self.input_branch, 
                             main_stream=self.main_stream, num_class=self.num_class,
                             reduction=self.reduction, block=self.block,graph=self.A)

    def forward(self, inputs):

        ipts, labs, type_, view_, seqL = inputs
        x_input = ipts[0] 
        N, T, V, I, C = x_input.size()
        pose  = x_input
        flip_idx = self.graph.flip_idx

        if not self.training and self.tta:
            multi_input = MultiInput(self.graph.connect_joint, self.graph.center)
            x1 = []
            x2 = []
            for i in range(N):
                x1.append(multi_input(x_input[i,:,:,0,:3].flip(0)))
                x2.append(multi_input(x_input[i,:,flip_idx,0,:3]))
            x_input = torch.cat([x_input, torch.stack(x1,0), torch.stack(x2,0)], dim=0)
        
        x = x_input.permute(0, 3, 4, 1, 2).contiguous()

        # resgcn
        x = self.ResGCN(x)

        if not self.training and self.tta:
            f1, f2, f3 = torch.split(x, [N, N, N], dim=0)
            x = torch.cat((f1, f2, f3), dim=1)
             
        embed = torch.unsqueeze(x,-1)
        
        retval = {
            'training_feat': {
                'SupConLoss': {'features': x , 'labels': labs}, # loss
            },
            'visual_summary': {
                'image/pose': pose.view(N*T, 1, I*V, C).contiguous() # visualization
            },
            'inference_feat': {
                'embeddings': embed # for metric
            }
        }
        return retval
    
class MultiInput:
    def __init__(self, connect_joint, center):
        self.connect_joint = connect_joint
        self.center = center

    def __call__(self, data):

        # T, V, C -> T, V, I=3, C + 2
        T, V, C = data.shape
        x_new = torch.zeros((T, V, 3, C + 2), device=data.device)

        # Joints
        x = data
        x_new[:, :, 0, :C] = x
        for i in range(V):
            x_new[:, i, 0, C:] = x[:, i, :2] - x[:, self.center, :2]

        # Velocity
        for i in range(T - 2):
            x_new[i, :, 1, :2] = x[i + 1, :, :2] - x[i, :, :2]
            x_new[i, :, 1, 3:] = x[i + 2, :, :2] - x[i, :, :2]
        x_new[:, :, 1, 3] = x[:, :, 2]

        # Bones
        for i in range(V):
            x_new[:, i, 2, :2] = x[:, i, :2] - x[:, self.connect_joint[i], :2]
        bone_length = 0
        for i in range(C - 1):
            bone_length += torch.pow(x_new[:, :, 2, i], 2)
        bone_length = torch.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            x_new[:, :, 2, C+i] = torch.acos(x_new[:, :, 2, i] / bone_length)
        x_new[:, :, 2, 3] = x[:, :, 2]

        data = x_new
        return data

