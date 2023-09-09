# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 21847 --nproc_per_node=4 opengait/main.py --cfgs /home/jdy/OpenGaitPose/configs/gaitgraph1/gaitgraph1_phase1_OUMVLP17.yaml --phase train  --log_to_file
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 22746 --nproc_per_node=4 opengait/main.py --cfgs /home/jdy/OpenGaitPose/configs/gaitgraph2/gaitgraph2_Gait3D.yaml --phase train --log_to_file
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 21847 --nproc_per_node=4 opengait/main.py --cfgs /home/jdy/OpenGaitPose/configs/gaittr/gaittr_OUMVLP.yaml --phase train  --log_to_file
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1247 opengait/main.py --cfgs /home/jdy/OpenGaitPose/configs/msgg/msgg_OUMVLP.yaml --phase train --log_to_file

