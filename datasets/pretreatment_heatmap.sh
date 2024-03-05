CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_P2P_DISABLE=1 \
python -m torch.distributed.launch \
--nproc_per_node=8 datasets/pretreatment_heatmap.py \
--pose_data_path=/home/mjz/Python_workspace/OpenGait/datasets/Gait3D/Gait3D_pose_pkl \
--save_root=/data3/gait_heatmap_data/Gait3D/ \
--ext_name=base \
--dataset_name=gait3d \
--heatemap_cfg_path=configs/skeletongait/pretreatment_heatmap.yaml