import re
import os
import json
import logging
import argparse
import pickle as pk
from typing import Tuple
from pathlib import Path
from functools import partial
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

"""
This script tries to match all the potential poses detected in a frame with the silhouette of the same frame in OUMVLP dataset,
and selects the pose that best matches the silhouette as the final pose for that frame, save its index in a pickle file which 
is used when extracting pose pkls.

More info please refer to https://github.com/ShiqiYu/OpenGait/pull/280
"""

def pose_silu_match_score(pose: np.ndarray, silu: np.ndarray) -> float:
    """
    Calculate the matching score between a 2D pose and a silhouette image using the sum of all joints' pixel intensity.

    Args:
        pose (np.ndarray): 2D pose, shape (n_joints, 3)
        silu (np.ndarray): silhouette image, shape (H, W, 3)

    Returns:
        float: matching score
    """
    pose_coord = pose[:,:2].astype(np.int32)
    
    H, W, *_ = silu.shape
    valid_joints = (pose_coord[:, 1] >=0) & (pose_coord[:, 1] < H) & \
                    (pose_coord[:, 0] >=0) & (pose_coord[:, 0] < W)

    if np.sum(valid_joints) == len(pose_coord):
        # only calculate score for points that are inside the silu img
        # use the sum of all joints' pixel intensity as the score
        return np.sum(silu[pose_coord[:, 1], pose_coord[:, 0]])
    else:
        # if pose coord is out of bound, return -inf
        return -np.inf


def perseq_pipeline(txt_groups: Tuple, rearrange_silu_root: Path, output_path: Path, verbose: bool = False) -> None:
    """
    Generate and save the pose selection index pickle file for a given sequence.

    Args:
        txt_groups (Tuple): Tuple of (sid, seq, view) and list of pose json paths.
        rearrange_silu_root (Path): Root dir of rearranged silu dataset.
        output_path (Path): Output path.
        verbose (bool, optional): Display debug info. Defaults to False.
    """    

    # resolve seq info
    sinfo = txt_groups[0]
    txt_paths = txt_groups[1]
    pick_idx = dict()

    # prepare output dir & resume last work
    dst_path = os.path.join(output_path, *sinfo)
    os.makedirs(dst_path, exist_ok=True)
    pkl_path = os.path.join(dst_path, 'pose_selection_idx.pkl')
    if os.path.exists(pkl_path):
        logging.debug(f'Pose index file {pkl_path} already exists, skipping...')
        return

    # extract
    for txt_file in sorted(txt_paths):
        # get the frame index (digit str before extension) of current frame
        try:
            frame_idx = re.findall(r'(\d+).json', os.path.basename(txt_file))[0]
        except IndexError:
            # adapt to different name format for json files in ID 00001
            frame_idx = re.findall(r'\d{4}', os.path.basename(txt_file))[0]

        with open(txt_file) as f:
            jsondata = json.load(f)

        person_num = len(jsondata['people'])

        # if no person or 1 person detected in this frame
        # we don't need to do the matching, just use the first or skip this frame when extracting pose pkl
        # see datasets/pretreatment.py#Line: 167~168 and Line: 173
        if person_num <= 1:
            continue

        # multiple people detected in this frame
        else:
            # load the reference silu image
            img_name = f'{frame_idx}.png'
            img_path = os.path.join(rearrange_silu_root, *sinfo, img_name)
            if not os.path.exists(img_path):
                logging.warning(
                    f'Pose reference silu({img_path}) of seq({'-'.join(sinfo)}) not exists, the matching for frame {frame_idx} is skipped. '
                    + 'This means that the first person in the frame will be used as the pose data, and this may cause performance degradation.'
                )
                continue
            silu_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # determine which pose has the highest matching score
            person_poses = [np.array(p["pose_keypoints_2d"]).reshape(-1,3) for p in jsondata['people']]
            max_score_idx = np.argmax([pose_silu_match_score(p, silu_img) for p in person_poses])
            
            # use the pose with the highest matching score to be the pkl data
            pick_idx[frame_idx] = max_score_idx
    
    # dump the index dict
    if verbose:
        logging.debug(f'Saving {pkl_path}... ')
    with open(pkl_path, 'wb') as f:
        pk.dump(pick_idx, f)
    logging.debug(f'Saved {len(pick_idx)} indexs to {pkl_path}.')


def main(rearrange_pose_root: Path, rearrange_silu_root: Path, output_path: Path, workers: int = 4, verbose: bool = False) -> None:
    """Reads a dataset and saves the data in pickle format.

    Args:
        rearrange_pose_root (Path): Root path of the rearranged oumvlp pose dataset.
        rearrange_silu_root (Path): Root path of the rearranged oumvlp silu dataset.
        output_path (Path): The selection index output path. The final structure is: output_path/sid/seq/view/pose_selection_idx.pkl
        workers (int, optional): Number of thread workers. Defaults to 4.
        verbose (bool, optional): Display debug info. Defaults to False.
    """
    txt_groups = defaultdict(list)
    logging.info(f'Listing {rearrange_pose_root}')
    total_files = 0

    for json_path in rearrange_pose_root.rglob('*.json'):
        if verbose:
            logging.debug(f'Adding {json_path}')
        *_, sid, seq, view, _ = json_path.as_posix().split(os.path.sep)
        txt_groups[(sid, seq, view)].append(json_path)
        total_files += 1

    logging.info(f'Total files listed: {total_files}')

    progress = tqdm(total=len(txt_groups), desc='Extracting Matching Pose Index', unit='seq')

    with mp.Pool(workers) as pool:
        logging.info(f'Start extracting pose indexes for {rearrange_pose_root}')
        for _ in pool.imap_unordered(
            partial(perseq_pipeline, rearrange_silu_root=rearrange_silu_root, output_path=output_path, verbose=verbose), 
            txt_groups.items()
        ):
            progress.update(1)
    
    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OUMVLP pose selection index extraction module.')
    parser.add_argument('-p', '--rearrange_pose_root', required=True, type=str, help='Root path of the rearranged oumvlp pose dataset.')
    parser.add_argument('-s', '--rearrange_silu_root', required=True, type=str, help='Root path of the rearranged oumvlp silu dataset.')
    parser.add_argument('-o', '--output_path', required=True, type=str, help='Output path of pickled dataset.')
    parser.add_argument('-l', '--log_file', default='./pretreatment.log', type=str, help='Log file path. Default: ./pretreatment.log')
    parser.add_argument('-n', '--n_workers', default=4, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Display debug info.')
    args = parser.parse_args()

    # logging and verbose mode
    logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w', format='[%(asctime)s - %(levelname)s]: %(message)s')
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Verbose mode is on.')
        for k, v in args.__dict__.items():
            logging.debug(f'{k}: {v}')

    # arguments validation
    args.rearrange_pose_root = os.path.abspath(args.rearrange_pose_root)
    assert os.path.exists(args.rearrange_pose_root), f"The specified oumvlp pose root({args.rearrange_pose_root}) does not exist."

    args.rearrange_silu_root = os.path.abspath(args.rearrange_silu_root)
    assert os.path.exists(args.rearrange_silu_root), f"The specified oumvlp silu root({args.rearrange_silu_root}) does not exist."

    args.output_path = os.path.abspath(args.output_path)
    os.makedirs(args.output_path, exist_ok=True)
    
    # run
    main(
        rearrange_pose_root=Path(args.rearrange_pose_root), 
        rearrange_silu_root=Path(args.rearrange_silu_root),
        output_path=Path(args.output_path), 
        workers=args.n_workers, 
        verbose=args.verbose
    )
