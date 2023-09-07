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

def txts2pickle(txt_groups: Tuple, output_path: Path, verbose: bool = False, dataset='CASIAB') -> None:
    """
    Reads a group of images and saves the data in pickle format.

    Args:
        img_groups (Tuple): Tuple of (sid, seq, view) and list of image paths.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        verbose (bool, optional): Display debug info. Defaults to False.
    """    
    
    sinfo = txt_groups[0]
    txt_paths = txt_groups[1]
    to_pickle = []
    if dataset == 'OUMVLP':
        for txt_file in sorted(txt_paths):
            try:
                with open(txt_file) as f:
                    jsondata = json.load(f)
                if len(jsondata['people'])==0:
                    continue
                data = np.array(jsondata["people"][0]["pose_keypoints_2d"]).reshape(-1,3)
                to_pickle.append(data)
            except:
                print(txt_file)
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



def pretreat(input_path: Path, output_path: Path, workers: int = 4, verbose: bool = False, dataset='CASIAB') -> None:
    """Reads a dataset and saves the data in pickle format.

    Args:
        input_path (Path): Dataset root path.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        workers (int, optional): Number of thread workers. Defaults to 4.
        verbose (bool, optional): Display debug info. Defaults to False.
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
        for _ in pool.imap_unordered(partial(txts2pickle, output_path=output_path, verbose=verbose, dataset=args.dataset), txt_groups.items()):
            progress.update(1)
    logging.info('Done')


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGait dataset pretreatment module.')
    parser.add_argument('-i', '--input_path', default='', type=str, help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='', type=str, help='Output path of pickled dataset.')
    parser.add_argument('-l', '--log_to_file', default='./pretreatment.log', type=str, help='Log file path. Default: ./pretreatment.log')
    parser.add_argument('-n', '--n_workers', default=4, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument('-d', '--dataset', default='CASIAB', type=str, help='Dataset for pretreatment.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Display debug info.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_to_file, filemode='w', format='[%(asctime)s - %(levelname)s]: %(message)s')
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Verbose mode is on.')
        for k, v in args.__dict__.items():
            logging.debug(f'{k}: {v}')

    pretreat(input_path=Path(args.input_path), output_path=Path(args.output_path), workers=args.n_workers, verbose=args.verbose, dataset=args.dataset)

    