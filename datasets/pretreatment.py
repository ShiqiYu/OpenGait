# This source is based on https://github.com/AbnerHqC/GaitSet/blob/master/pretreatment.py
import argparse
import logging
import multiprocessing as mp
import os
import re
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
import json

def imgs2pickle(img_groups: Tuple, output_path: Path, img_size: int = 64, verbose: bool = False, dataset='CASIAB') -> None:
    """Reads a group of images and saves the data in pickle format.

    Args:
        img_groups (Tuple): Tuple of (sid, seq, view) and list of image paths.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        verbose (bool, optional): Display debug info. Defaults to False.
    """    
    sinfo = img_groups[0]
    img_paths = img_groups[1]
    to_pickle = []
    for img_file in sorted(img_paths):
        if verbose:
            logging.debug(f'Reading sid {sinfo[0]}, seq {sinfo[1]}, view {sinfo[2]} from {img_file}')

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        
        if dataset == 'GREW':
            to_pickle.append(img.astype('uint8'))
            continue

        if img.sum() <= 10000:
            if verbose:
                logging.debug(f'Image sum: {img.sum()}')
            logging.warning(f'{img_file} has no data.')
            continue

        # Get the upper and lower points
        y_sum = img.sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
        img = img[y_top: y_btm + 1, :]

        # As the height of a person is larger than the width,
        # use the height to calculate resize ratio.
        ratio = img.shape[1] / img.shape[0]
        img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

        # Get the median of the x-axis and take it as the person's x-center.
        x_csum = img.sum(axis=0).cumsum()
        x_center = None
        for idx, csum in enumerate(x_csum):
            if csum > img.sum() / 2:
                x_center = idx
                break

        if not x_center:
            logging.warning(f'{img_file} has no center.')
            continue

        # Get the left and right points
        half_width = img_size // 2
        left = x_center - half_width
        right = x_center + half_width
        if left <= 0 or right >= img.shape[1]:
            left += half_width
            right += half_width
            _ = np.zeros((img.shape[0], half_width))
            img = np.concatenate([_, img, _], axis=1)

        to_pickle.append(img[:, left: right].astype('uint8'))

    if to_pickle:
        to_pickle = np.asarray(to_pickle)
        dst_path = os.path.join(output_path, *sinfo)
        # print(img_paths[0].as_posix().split('/'),img_paths[0].as_posix().split('/')[-5])
        # dst_path = os.path.join(output_path, img_paths[0].as_posix().split('/')[-5], *sinfo) if dataset == 'GREW' else dst
        os.makedirs(dst_path, exist_ok=True)
        pkl_path = os.path.join(dst_path, f'{sinfo[2]}.pkl')
        if verbose:
            logging.debug(f'Saving {pkl_path}...')
        pickle.dump(to_pickle, open(pkl_path, 'wb'))   
        logging.info(f'Saved {len(to_pickle)} valid frames to {pkl_path}.')


    if len(to_pickle) < 5:
        logging.warning(f'{sinfo} has less than 5 valid data.')



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
    for img_path in input_path.rglob('*.png'):
        if 'gei.png' in img_path.as_posix():
            continue
        if verbose:
            logging.debug(f'Adding {img_path}')
        *_, sid, seq, view, _ = img_path.as_posix().split('/')
        img_groups[(sid, seq, view)].append(img_path)
        total_files += 1

    logging.info(f'Total files listed: {total_files}')

    progress = tqdm(total=len(img_groups), desc='Pretreating', unit='folder')

    with mp.Pool(workers) as pool:
        logging.info(f'Start pretreating {input_path}')
        for _ in pool.imap_unordered(partial(imgs2pickle, output_path=output_path, img_size=img_size, verbose=verbose, dataset=dataset), img_groups.items()):
            progress.update(1)
    logging.info('Done')

def txts2pickle(txt_groups: Tuple, output_path: Path, verbose: bool = False, dataset='CASIAB', **kwargs) -> None:
    """
    Reads a group of images and saves the data in pickle format.

    Args:
        txt_groups (Tuple): Tuple of (sid, seq, view) and list of image paths.
        output_path (Path): Output path.
        verbose (bool, optional): Display debug info. Defaults to False.
        dataset (str, optional): Dataset name. Defaults to 'CASIAB'.
        kwargs (dict, optional): Additional arguments. It receives 'oumvlp_index_dir' when dataset is 'OUMVLP'.
    """    

    sinfo = txt_groups[0]
    txt_paths = txt_groups[1]
    to_pickle = []
    if dataset == 'OUMVLP':
        # load pose selection index
        idx_file = os.path.join(kwargs['oumvlp_index_dir'], *sinfo, 'pose_selection_idx.pkl')
        try:
            with open(idx_file, 'rb') as f:
                frame_wise_idx = pickle.load(f) # dict, structure is {txt_file_name(str): selected_pose_idx(int)}
        except FileNotFoundError:
            logging.warning(
                f'No pose selection index found for sequence: {sinfo}, will use the first detected pose for each frame. '
                + 'This may cause performance degradation, see https://github.com/ShiqiYu/OpenGait/pull/280 for more details. '
                + 'You can avoid this warning by re-get the index files following Step4-2 in datasets/OUMVLP/README.md.'
            )
            frame_wise_idx = dict()

        # apply selection index for each frame in current sequence
        for txt_file in sorted(txt_paths):
            try:
                with open(txt_file) as f:
                    jsondata = json.load(f)

                # no person detected in this frame
                if len(jsondata['people'])==0:
                    continue
                
                # get the frame index (digit str before extension) of current frame
                try:
                    frame_idx = re.findall(r'(\d+).json', os.path.basename(txt_file))[0]
                except IndexError:
                    # adapt to different name format for json files in ID 00001
                    frame_idx = re.findall(r'\d{4}', os.path.basename(txt_file))[0]

                # use the first person if no index file or less than one pose in current frame
                pose_idx = frame_wise_idx.get(frame_idx, 0) 

                data = np.array(jsondata["people"][pose_idx]["pose_keypoints_2d"]).reshape(-1,3)
                to_pickle.append(data)
            except:
                print(f"Fail to extract pkl for frame({txt_file}), seq({sinfo}).")
    else:
        for txt_file in sorted(txt_paths):
            if verbose:
                logging.debug(f'Reading sid {sinfo[0]}, seq {sinfo[1]}, view {sinfo[2]} from {txt_file}')
            data = np.genfromtxt(txt_file, delimiter=',')[2:].reshape(-1,3)
            to_pickle.append(data)
        
    if to_pickle:
        dst_path = os.path.join(output_path, *sinfo)
        keypoints = np.stack(to_pickle)
        os.makedirs(dst_path, exist_ok=True)
        pkl_path = os.path.join(dst_path, f'{sinfo[2]}.pkl')
        if verbose:
            logging.debug(f'Saving {pkl_path}...')
        pickle.dump(keypoints, open(pkl_path, 'wb'))   
        logging.info(f'Saved {len(to_pickle)} valid frames\' keypoints to {pkl_path}.')

    if len(to_pickle) < 5:
        logging.warning(f'{sinfo} has less than 5 valid data.')


def pretreat_pose(input_path: Path, output_path: Path, workers: int = 4, verbose: bool = False, dataset='CASIAB', **kwargs) -> None:
    """Reads a dataset and saves the data in pickle format.

    Args:
        input_path (Path): Dataset root path.
        output_path (Path): Output path.
        workers (int, optional): Number of thread workers. Defaults to 4.
        verbose (bool, optional): Display debug info. Defaults to False.
        dataset (str, optional): Dataset name. Defaults to 'CASIAB'.
        kwargs (dict, optional): Additional arguments. It receives 'oumvlp_index_dir' when dataset is 'OUMVLP'.
    """
    txt_groups = defaultdict(list)
    logging.info(f'Listing {input_path}')
    total_files = 0
    if dataset == 'OUMVLP':
        for json_path in input_path.rglob('*.json'):
            if verbose:
                logging.debug(f'Adding {json_path}')
            *_, sid, seq, view, _ = json_path.as_posix().split('/')
            txt_groups[(sid, seq, view)].append(json_path)
            total_files += 1
    else:
        for txt_path in input_path.rglob('*.txt'):
            if verbose:
                logging.debug(f'Adding {txt_path}')
            *_, sid, seq, view, _ = txt_path.as_posix().split('/')
            txt_groups[(sid, seq, view)].append(txt_path)
            total_files += 1

    logging.info(f'Total files listed: {total_files}')

    progress = tqdm(total=len(txt_groups), desc='Pretreating', unit='folder')

    with mp.Pool(workers) as pool:
        logging.info(f'Start pretreating {input_path}')
        for _ in pool.imap_unordered(
            partial(txts2pickle, output_path=output_path, verbose=verbose, dataset=args.dataset, **kwargs), 
            txt_groups.items()
        ):
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
    parser.add_argument('-p', '--pose', default=False, action='store_true', help='Processing pose.')
    parser.add_argument('-oid', '--oumvlp_index_dir', default='', type=str, 
                        help='Path of the directory containing all index files for extracting oumvlp pose pkl, which is necessary to promise the temporal consistency of extracted pose sequence. ' 
                        + 'Note: this argument is only used when extracting oumvlp pose pkl, more info please refer to Step4-2 in datasets/OUMVLP/README.md. ')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w', format='[%(asctime)s - %(levelname)s]: %(message)s')
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Verbose mode is on.')
        for k, v in args.__dict__.items():
            logging.debug(f'{k}: {v}')
    if args.pose:
        if args.dataset.lower() == "oumvlp":
            assert args.oumvlp_index_dir, (
                "When extracting the oumvlp pose pkl, please specify the path of the directory containing all index files using the `--oumvlp_index_dir` argument."
                + "If you don't know what it is, please refer to Step4-2 in datasets/OUMVLP/README.md for more details."
            )
            
            args.oumvlp_index_dir = os.path.abspath(args.oumvlp_index_dir)
            assert os.path.exists(args.oumvlp_index_dir), f"The specified oumvlp index files' directory({args.oumvlp_index_dir}) does not exist."
            
            logging.info(f'Using the oumvlp index files in {args.oumvlp_index_dir}')
        
        pretreat_pose(
            input_path=Path(args.input_path), 
            output_path=Path(args.output_path), 
            workers=args.n_workers, 
            verbose=args.verbose, 
            dataset=args.dataset,
            oumvlp_index_dir=args.oumvlp_index_dir
        )
    else:
        pretreat(input_path=Path(args.input_path), output_path=Path(args.output_path), img_size=args.img_size, workers=args.n_workers, verbose=args.verbose, dataset=args.dataset)
