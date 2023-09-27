import torch
from ..base_model import BaseModel
from ..backbones.resgcn import ResGCN
from ..modules import Graph
import torch.nn.functional as F

class GaitGraph1(BaseModel):
    """
        GaitGraph1: Gaitgraph: Graph Convolutional Network for Skeleton-Based Gait Recognition
        Paper:    https://ieeexplore.ieee.org/document/9506717
        Github:   https://github.com/tteepe/GaitGraph
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
        x_input = ipts[0] # N T C V I
        # x = N, T, C, V, M -> N, C, T, V, M
        x_input = x_input.permute(0, 2, 3, 4, 1).contiguous()
        N, T, V, I, C = x_input.size() 
        
        pose  = x_input
        if self.training:
            x_input = torch.cat([x_input[:,:int(T/2),...],x_input[:,int(T/2):,...]],dim=0) #[8, 60, 17, 1, 3]
        elif self.tta:
            data_flipped = torch.flip(x_input,dims=[1])
            x_input = torch.cat([x_input,data_flipped], dim=0)

        x = x_input.permute(0, 3, 4, 1, 2).contiguous()

        # resgcn
        x = self.ResGCN(x)
        x = F.normalize(x, dim=1, p=2) # norm #only for GaitGraph1 # Remove from GaitGraph2
        
        if self.training:
            f1, f2 = torch.split(x, [N, N], dim=0)
            embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) #[4, 2, 128]
            
        elif self.tta:
            f1, f2 = torch.split(x, [N, N], dim=0)
            embed = torch.mean(torch.stack([f1, f2]), dim=0)
            embed = embed.unsqueeze(-1)
        else:
            embed = embed.unsqueeze(-1)
        
        retval = {
            'training_feat': {
                'SupConLoss': {'features': embed , 'labels': labs}, # loss
            },
            'visual_summary': {
                'image/pose': pose.view(N*T, 1, I*V, C).contiguous() # visualization
            },
            'inference_feat': {
                'embeddings':   embed # for metric
            }
        }
        return retval
