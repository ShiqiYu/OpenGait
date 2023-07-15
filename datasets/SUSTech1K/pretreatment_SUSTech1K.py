# This source is based on https://github.com/AbnerHqC/GaitSet/blob/master/pretreatment.py
import argparse
import logging
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

import json
import open3d as o3d

def compare_pcd_rgb_timestamp(pcd_file,rgb_file):
    pcd_time = float(pcd_file.split('/')[-1].replace('.pcd','')) + 0.05
    rgb_time = float(rgb_file.split('/')[-1].replace('.jpg','')[:10] + '.' + rgb_file.split('/')[-1].replace('.jpg','')[10:])
    return pcd_time, rgb_time



def imgs2pickle(img_groups: Tuple, output_path: Path, img_size: int = 64, verbose: bool = False, dataset='CASIAB') -> None:
    """Reads a group of images and saves the data in pickle format.

    Args:
        img_groups (Tuple): Tuple of (sid, seq, view) and list of image paths.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        verbose (bool, optional): Display debug info. Defaults to False.
    """    
    sinfo = img_groups[0]
    img_paths = img_groups[1] # path with modality name
    to_pickle = []
    cnt = 0
    pcd_list = []
    rgb_list = []

    threshold = 0.020 # 20 ms

    for index, modality_files in enumerate(img_paths):
        data_files = modality_files[1]
        modality = modality_files[0]
        if modality == 'PCDs':
            data = [np.asarray(o3d.io.read_point_cloud(points).points) for points in data_files]
            pcd_list = data_files
        elif modality == 'RGB_raw':
            imgs = [cv2.imread(rgb) for rgb in data_files]
            rgb_list = data_files
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
            HWs =  [img.shape[:2] for img in imgs]
            # transpose to (C, H W)
            data = [cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC) for img in imgs]
            imgs = [img.transpose(2, 0, 1) for img in imgs]
            data = np.asarray(data)
            HWs = np.asarray(HWs)
        elif modality == 'Sils_raw':
            sils = [cv2.imread(sil, cv2.IMREAD_GRAYSCALE) for sil in data_files]
            data = [cv2.resize(sil, (img_size, img_size), interpolation=cv2.INTER_CUBIC) for sil in sils]
            data = np.asarray(data)
        elif modality == 'Sils_aligned':
            sils = [cv2.imread(sil, cv2.IMREAD_GRAYSCALE) for sil in data_files]
            data = [cv2.resize(sil, (img_size, img_size), interpolation=cv2.INTER_CUBIC) for sil in sils]
            data = np.asarray(data)
        elif modality == 'Pose':
            data = [json.load(open(pose)) for pose in data_files]
            data = np.asarray(data)
        elif modality == 'PCDs_depths':
            imgs = [cv2.imread(rgb) for rgb in data_files]
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
            data = [img.transpose(2, 0, 1) for img in imgs]       
            data = np.asarray(data)
        elif modality == 'PCDs_sils':
            data = [cv2.imread(sil, cv2.IMREAD_GRAYSCALE) for sil in data_files]
            data = np.asarray(data)

        dst_path = os.path.join(output_path, *sinfo)
        os.makedirs(dst_path, exist_ok=True)
        if modality == 'RGB_raw':
            pkl_path = os.path.join(dst_path, f'{cnt:02d}-{sinfo[2]}-Camera-Ratios-HW.pkl')
            pickle.dump(HWs, open(pkl_path, 'wb'))   
            cnt += 1

        if 'PCDs' in modality:
            pkl_path = os.path.join(dst_path, f'{cnt:02d}-{sinfo[2]}-LiDAR-{modality}.pkl')
            pickle.dump(data, open(pkl_path, 'wb'))   
        else:
            pkl_path = os.path.join(dst_path, f'{cnt:02d}-{sinfo[2]}-Camera-{modality}.pkl')
            pickle.dump(data, open(pkl_path, 'wb'))   
        cnt += 1

    pcd_indexs = []
    rgb_indexs = []
    # print(pcd_list)
    for pcd_index in range(len(pcd_list)):
        time_diff = 1
        tmp = pcd_index, 0
        for rgb_index in range(len(rgb_list)):
            pcd_t, rgb_t = compare_pcd_rgb_timestamp(pcd_list[pcd_index], rgb_list[rgb_index])
            diff = abs(pcd_t - rgb_t)
            if diff < time_diff:
                tmp = pcd_index, rgb_index
                time_diff = diff
        if time_diff <= threshold:
            pcd_indexs.append(tmp[0])
            rgb_indexs.append(tmp[1])
            
    if len(set(pcd_indexs)) != len(pcd_indexs):
        print(img_groups[0], pcd_indexs, rgb_indexs, len(pcd_indexs) == len(pcd_indexs))

    for index, modality_files in enumerate(img_paths):
        modality = modality_files[0]
        data_files = modality_files[1]
        data_files = [data_files[index] for index in pcd_indexs] if 'PCDs' in modality else [data_files[index] for index in rgb_indexs]

        if modality == 'PCDs':
            data = [np.asarray(o3d.io.read_point_cloud(points).points) for points in data_files]
            pcd_list = data_files
        elif modality == 'RGB_raw':
            imgs = [cv2.imread(rgb) for rgb in data_files]
            rgb_list = data_files
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
            HWs =  [img.shape[:2] for img in imgs]
            # transpose to (C, H W)
            data = [cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC) for img in imgs]
            imgs = [img.transpose(2, 0, 1) for img in imgs]
            data = np.asarray(data)
            HWs = np.asarray(HWs)
        elif modality == 'Sils_raw':
            sils = [cv2.imread(sil, cv2.IMREAD_GRAYSCALE) for sil in data_files]
            data = [cv2.resize(sil, (img_size, img_size), interpolation=cv2.INTER_CUBIC) for sil in sils]
            data = np.asarray(data)
        elif modality == 'Sils_aligned':
            sils = [cv2.imread(sil, cv2.IMREAD_GRAYSCALE) for sil in data_files]
            data = [cv2.resize(sil, (img_size, img_size), interpolation=cv2.INTER_CUBIC) for sil in sils]
            data = np.asarray(data)
        elif modality == 'Pose':
            data = [json.load(open(pose)) for pose in data_files]
            data = np.asarray(data)
        elif modality == 'PCDs_depths':
            imgs = [cv2.imread(rgb) for rgb in data_files]
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
            data = [img.transpose(2, 0, 1) for img in imgs]       
            data = np.asarray(data)
        elif modality == 'PCDs_sils':
            data = [cv2.imread(sil, cv2.IMREAD_GRAYSCALE) for sil in data_files]
            data = np.asarray(data)

        dst_path = os.path.join(output_path, *sinfo)
        os.makedirs(dst_path, exist_ok=True)
        if modality == 'RGB_raw':
            pkl_path = os.path.join(dst_path, f'{cnt:02d}-sync-{sinfo[2]}-Camera-Ratios-HW.pkl')
            pickle.dump(HWs, open(pkl_path, 'wb'))   
            cnt += 1

        if 'PCDs' in modality:
            pkl_path = os.path.join(dst_path, f'{cnt:02d}-sync-{sinfo[2]}-LiDAR-{modality}.pkl')
            pickle.dump(data, open(pkl_path, 'wb'))   
        else:
            pkl_path = os.path.join(dst_path, f'{cnt:02d}-sync-{sinfo[2]}-Camera-{modality}.pkl')
            pickle.dump(data, open(pkl_path, 'wb'))   
        cnt += 1


def pretreat(input_path: Path, output_path: Path, img_size: int = 64, workers: int = 4, verbose: bool = False, dataset: str = 'CASIAB') -> None:
    """Reads a dataset and saves the data in pickle format.

    Args:
        input_path (Path): Dataset root path.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        workers (int, optional): Number of thread workers. Defaults to 4.
        verbose (bool, optional): Display debug info. Defaults to False.
    """
    img_groups = defaultdict(list)
    logging.info(f'Listing {input_path}')
    total_files = 0
    for id_ in tqdm(sorted(os.listdir(input_path))):    
        for type_ in os.listdir(os.path.join(input_path,id_)):
            for view_ in os.listdir(os.path.join(input_path,id_,type_)):
                for modality in sorted(os.listdir(os.path.join(input_path,id_,type_,view_))):
                    modality_path = os.path.join(input_path,id_,type_,view_,modality)
                    file_names = sorted(os.listdir(modality_path))
                    file_names = [os.path.join(modality_path, file_name) for file_name in file_names]
                    img_groups[(id_, type_, view_)].append((modality, file_names))
                    total_files += 1

    logging.info(f'Total files listed: {total_files}')

    progress = tqdm(total=len(img_groups), desc='Pretreating', unit='folder')

    with mp.Pool(workers) as pool:
        logging.info(f'Start pretreating {input_path}')
        for _ in pool.imap_unordered(partial(imgs2pickle, output_path=output_path, img_size=img_size, verbose=verbose, dataset=dataset), img_groups.items()):
            progress.update(1)
    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGait dataset pretreatment module.')
    parser.add_argument('-i', '--input_path', default='', type=str, help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='', type=str, help='Output path of pickled dataset.')
    parser.add_argument('-l', '--log_file', default='./pretreatment.log', type=str, help='Log file path. Default: ./pretreatment.log')
    parser.add_argument('-n', '--n_workers', default=4, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument('-r', '--img_size', default=64, type=int, help='Image resizing size. Default 64')
    parser.add_argument('-d', '--dataset', default='CASIAB', type=str, help='Dataset for pretreatment.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Display debug info.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w', format='[%(asctime)s - %(levelname)s]: %(message)s')
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Verbose mode is on.')
        for k, v in args.__dict__.items():
            logging.debug(f'{k}: {v}')

    pretreat(input_path=Path(args.input_path), output_path=Path(args.output_path), img_size=args.img_size, workers=args.n_workers, verbose=args.verbose, dataset=args.dataset)