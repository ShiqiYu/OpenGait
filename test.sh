# # **************** For CASIA-B ****************
# # Baseline
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_da_grew.yaml --phase test --log_to_file

# # GaitSet
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gaitset/gaitset.yaml --phase test

# # GaitPart
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gaitpart/gaitpart.yaml --phase test

# GaitGL
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgl/gaitgl.yaml --phase test

# # GLN 
# # Phase 1
# CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --master_port 12345  --nproc_per_node=2 opengait/main.py --cfgs ./configs/gln/gln_phase1.yaml --phase test
# # Phase 2
# CUDA_VISIBLE_DEVICES=2,5 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gln/gln_phase2.yaml --phase test


# # **************** For OUMVLP ****************
# # Baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/baseline/baseline_OUMVLP.yaml --phase test

# # GaitSet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitset/gaitset_OUMVLP.yaml --phase test

# # GaitPart
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_OUMVLP.yaml --phase test

# GaitGL
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_OUMVLP.yaml --phase test

# export NCCL_P2P_DISABLE=1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgraph1/gaitgraph1_phase1_Gait3D.yaml --phase test
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgraph1/gaitgraph1_phase1_OUMVLP.yaml --phase test

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 22335 --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgraph2/gaitgraph2_OUMVLP.yaml --phase test --log_to_file 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 22355 --nproc_per_node=4 opengait/main.py --cfgs /home/jdy/OpenGaitPose/configs/gaittr/gaittr_OUMVLP.yaml --phase test --log_to_file 
