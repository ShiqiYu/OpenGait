import os
import pickle
import shutil
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from multiprocessing import Pool

MODALITIES = ['ratios_HW', 'rgb_f', 'sil', 'par', 'pose']
RGB_SIZE = (128, 128)
BLACK_THRESHOLD = 15

def get_mod_indx(filename):
    for idx, modality in enumerate(MODALITIES):
        if modality in filename:
            return idx
    return None

def remove_padding_and_resize(img, threshold=0):
    """
    Remove padding and resize to RGB_SIZE

    Args:
        img (np.ndarray): Image to process.

    Returns:
        np.ndarray: Processed image.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # To better compute the mask, convert to grayscale
    mask = (img_gray > threshold)

    y_sum = mask.sum(axis=1)
    y_top = (y_sum != 0).argmax(axis=0)
    y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)

    x_sum = mask.sum(axis=0)
    x_top = (x_sum != 0).argmax(axis=0)
    x_btm = (x_sum != 0).cumsum(axis=0).argmax(axis=0)

    img = img[y_top:y_btm + 1, x_top:x_btm + 1]
    orig_size = list(img.shape[:2])
    img = cv2.resize(img, RGB_SIZE, interpolation=cv2.INTER_AREA)

    return img.astype('uint8'), orig_size

def load_rgb_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Load all frames in memory
    frames = []
    orig_sizes = [] 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, orig_size = remove_padding_and_resize(frame, threshold=BLACK_THRESHOLD)
        frames.append(frame)
        orig_sizes.append(orig_size)
    cap.release()

    # Convert list of frames to numpy array and change color space from BGR to RGB
    frames = np.array(frames)
    frames = frames[:, :, :, ::-1]  # Convert BGR to RGB
    frames = np.transpose(frames, (0, 3, 1, 2))  # Convert to (N, C, H, W)

    orig_sizes = np.array(orig_sizes)

    return frames, orig_sizes

def process_view(args_tuple):
    """Process a single (id, type, view) triplet. Designed to run in a worker process."""
    _id, _type, _view, sil_par_pose_path, rgb_path, output_path, copy_mode = args_tuple

    # save the RGB modality as a pkl if the RGB path is provided
    if rgb_path:
        rgb_filename = os.listdir(rgb_path)[0]  # Assuming there's only one video file in the directory
        rgb_seq, orig_sizes = load_rgb_video(os.path.join(rgb_path, rgb_filename))

        if rgb_seq.size == 0:
            print(f"Warning: No frames loaded from RGB video at {rgb_path}, skipping RGB modality for {_id}/{_type}/{_view}.")
        else:
            # Output RGB pkl filename
            pkl_filename = os.path.basename(rgb_filename).replace('.avi','') + '.pkl'
            mod_idx = get_mod_indx(pkl_filename)
            rgb_dst_path = os.path.join(output_path, _id, _type, _view, f'{mod_idx:03d}-{pkl_filename}')

            os.makedirs(os.path.dirname(rgb_dst_path), exist_ok=True)
            with open(rgb_dst_path, 'wb') as f:
                pickle.dump(rgb_seq, f)

            # Save original height and weight
            ratios_filename = os.path.basename(rgb_filename).replace('.avi','').replace('rgb_f','ratios_HW') + '.pkl'
            mod_idx = get_mod_indx(ratios_filename)
            ratios_hw_dst_path = os.path.join(output_path, _id, _type, _view, f'{mod_idx:03d}-{ratios_filename}')
            os.makedirs(os.path.dirname(ratios_hw_dst_path), exist_ok=True)
            with open(ratios_hw_dst_path, 'wb') as f:
                pickle.dump(orig_sizes, f)

    # Copy and move the pkl files for silhouette, parsing and pose data
    for pkl_filename in os.listdir(os.path.join(sil_par_pose_path, _id, _type, _view)):
        if pkl_filename.endswith('.pkl'):
            src_pkl_path = os.path.join(sil_par_pose_path, _id, _type, _view, pkl_filename)
            mod_idx = get_mod_indx(pkl_filename)
            if mod_idx is None:
                print(f"Warning: Could not determine modality for file {src_pkl_path}, skipping.")
                continue

            dst_pkl_path = os.path.join(output_path, _id, _type, _view, f'{mod_idx:03d}-{pkl_filename.replace(".avi", "")}')
            os.makedirs(os.path.dirname(dst_pkl_path), exist_ok=True)
            if copy_mode == 'copy':
                shutil.copy(src_pkl_path, dst_pkl_path)
            elif copy_mode == 'symlink':
                os.symlink(src_pkl_path, dst_pkl_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CCGR dataset Preprocessing.')
    parser.add_argument('--sil_par_pose_path', default='', type=str,
                        help='Root path of directory containing the silhouette, parsing and pose data.')
    parser.add_argument('--rgb_path', default='', type=str,
                        help='Root path of directory containing the RGB data (only for CCGR-Mini).')
    parser.add_argument('-o', '--output_path', default='',
                        type=str, help='Output path of pickled dataset.')
    parser.add_argument('--copy-mode', type=str, help='Copy mode for handling files.', choices=['copy', 'symlink'], required=False, default='copy')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of parallel worker processes (default: all CPU cores).')
    args = parser.parse_args()

    # Build the full list of (id, type, view) jobs
    print(f"Scanning sequences to process...")
    path = args.sil_par_pose_path
    jobs = []
    for _id in sorted(os.listdir(path)):
        id_path = os.path.join(path, _id)
        if not os.path.isdir(id_path):
            continue
        for _type in sorted(os.listdir(id_path)):
            type_path = os.path.join(id_path, _type)
            if not os.path.isdir(type_path):
                continue
            for _view in sorted(os.listdir(type_path)):

                if args.rgb_path:
                    rgb_path = os.path.join(args.rgb_path, _id, _type, _view)
                    if not os.path.isdir(rgb_path):
                        print(f"Warning: RGB path does not exist for {_id}/{_type}/{_view}, skipping.")
                        rgb_path = None
                else:
                    rgb_path = None

                jobs.append((_id, _type, _view,
                             path, rgb_path, args.output_path, args.copy_mode))

    print(f"Processing {len(jobs)} sequences with {args.num_workers} workers...")
    if args.num_workers > 1: 
        with Pool(processes=args.num_workers) as pool:
            list(tqdm(pool.imap_unordered(process_view, jobs), total=len(jobs)))
    else:
        for job in tqdm(jobs):
            process_view(job)
