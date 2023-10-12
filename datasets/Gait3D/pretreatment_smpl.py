# -*- coding: utf-8 -*-
"""
   Author :       jinkai Zheng
   date：          2021/10/30
   E-mail:        zhengjinkai3@qq.com
"""


import os.path as osp
import time
import os
import threading
import itertools
import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-i', '--input_path', default='', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('-o', '--output_path', default='', type=str,
                    help='Root path for output.')
opt = parser.parse_args()


def get_pickle(thread_id, id_list, save_dir):
    for id in sorted(id_list):
        print(f"Process threadID-PID: {thread_id}-{id}")
        cam_list = os.listdir(osp.join(data_dir, id))
        cam_list.sort()
        for cam in cam_list:
            seq_list = os.listdir(osp.join(data_dir, id, cam))
            seq_list.sort()
            for seq in seq_list:
                npz_list = os.listdir(osp.join(data_dir, id, cam, seq))
                npz_list.sort()
                smpl_paras_fras = []
                for npz in npz_list:
                    npz_path = osp.join(data_dir, id, cam, seq, npz)
                    frame = np.load(npz_path, allow_pickle=True)['results'][()][0]
                    smpl_cam = frame['cam']  # 3-D
                    smpl_pose = frame['poses']  # 72-D
                    smpl_shape = frame['betas']  # 10-D
                    smpl_paras = np.concatenate((smpl_cam, smpl_pose, smpl_shape), 0)
                    smpl_paras_fras.append(smpl_paras)
                smpl_paras_fras = np.asarray(smpl_paras_fras)

                out_dir = osp.join(save_dir, id, cam, seq)
                os.makedirs(out_dir)
                smpl_paras_fras_pkl = os.path.join(out_dir, '{}.pkl'.format(seq))
                pickle.dump(smpl_paras_fras, open(smpl_paras_fras_pkl, 'wb'))


if __name__ == '__main__':

    data_dir = opt.input_path

    save_dir = opt.output_path

    start_time = time.time()
    maxnum_thread = 8

    all_ids = sorted(os.listdir(data_dir))
    num_ids = len(all_ids)

    proces = []
    for thread_id in range(maxnum_thread):
        indices = itertools.islice(range(num_ids), thread_id, num_ids, maxnum_thread)
        id_list = [all_ids[i] for i in indices]
        thread_func = threading.Thread(target=get_pickle, args=(thread_id, id_list, save_dir))

        thread_func.start()
        proces.append(thread_func)

    for proc in proces:
        proc.join()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600,
        (time_elapsed - (time_elapsed // 3600) * 3600) // 60,
        time_elapsed % 60))