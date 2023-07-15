import matplotlib.pyplot as plt

import open3d as o3d
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

def align_img(img: np.ndarray, img_size: int = 64) -> np.ndarray:
    """Aligns the image to the center.
    Args:
        img (np.ndarray): Image to align.
        img_size (int, optional): Image resizing size. Defaults to 64.
    Returns:
        np.ndarray: Aligned image.
    """    
    if img.sum() <= 10000:
        y_top = 0
        y_btm = img.shape[0]
    else:
        # Get the upper and lower points
        # img.sum
        y_sum = img.sum(axis=2).sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)

    img = img[y_top: y_btm, :,:]

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)
    
    # Get the median of the x-axis and take it as the person's x-center.
    x_csum = img.sum(axis=2).sum(axis=0).cumsum()
    x_center = img.shape[1] // 2
    for idx, csum in enumerate(x_csum):
        if csum > img.sum() / 2:
            x_center = idx
            break

    # if not x_center:
    #     logging.warning(f'{img_file} has no center.')
    #     continue

    # Get the left and right points
    half_width = img_size // 2
    left = x_center - half_width
    right = x_center + half_width
    if left <= 0 or right >= img.shape[1]:
        left += half_width
        right += half_width
        # _ = np.zeros((img.shape[0], half_width,3))
        # img = np.concatenate([_, img, _], axis=1)
    
    img = img[:, left: right,:].astype('uint8')
    return img





def lidar_to_2d_front_view(points,
                           v_res,
                           h_res,
                           v_fov,
                           val="depth",
                           cmap="jet",
                           saveto=None,
                           y_fudge=0.0
                           ):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.

    Args:
        points: (np array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.

            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    assert len(v_fov) ==2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in {"depth", "height", "reflectance"}, \
        'val must be one of {"depth", "height", "reflectance"}'


    x_lidar = - points[:, 0]
    y_lidar = - points[:, 1]
    z_lidar = points[:, 2]
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_img -= x_min              # Shift
    x_max = 360.0 / h_res       # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res    # theoretical min y value based on sensor specs
    y_img -= y_min              # Shift
    y_max = v_fov_total / v_res # Theoretical max x value after shifting

    y_max += y_fudge            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pass
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -d_lidar
        # pixel_values = 'w'

    # PLOT THE IMAGE
    cmap = "jet"            # Color map to use
    dpi = 100               # Image resolution
    fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    ax.scatter(x_img,y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
    ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
    ax.axis('scaled')              # {equal, scaled}
    ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV

    saveto = saveto.replace('.pcd','.png')
    fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    img = cv2.imread(saveto)
    img = align_img(img)

    aligned_path = saveto.replace('offline','aligned')
    os.makedirs(os.path.dirname(aligned_path), exist_ok=True)
    cv2.imwrite(aligned_path, img)
    # fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    # ax.scatter(x_img,y_img, s=1, c='white', linewidths=0, alpha=1)
    # ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
    # ax.axis('scaled')              # {equal, scaled}
    # ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    # ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    # plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    # plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV

    # fig.savefig(saveto.replace('depth','sils'), dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    # plt.close()


def pcd2depth(img_groups: Tuple, output_path: Path, img_size: int = 64, verbose: bool = False, dataset='CASIAB') -> None:
    """Reads a group of images and saves the data in pickle format.
    Args:
        img_groups (Tuple): Tuple of (sid, seq, view) and list of image paths.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        verbose (bool, optional): Display debug info. Defaults to False.
    """    
    sinfo = img_groups[0]
    img_paths = img_groups[1]
    for img_file in sorted(img_paths):
        pcd_name = img_file.split('/')[-1]
        pcd = o3d.io.read_point_cloud(img_file)
        points = np.asarray(pcd.points)
        HRES = 0.19188        # horizontal resolution (assuming 20Hz setting)
        VRES = 0.2   
        VFOV = (-25.0, 15.0) # Field of view (-ve, +ve) along vertical axis
        Y_FUDGE = 0  # y fudge factor for velodyne HDL 64E
        dst_path = os.path.join(output_path, *sinfo)
        os.makedirs(dst_path, exist_ok=True)
        dst_path = os.path.join(dst_path,pcd_name)
        lidar_to_2d_front_view(points, v_res=VRES, h_res=HRES, v_fov=VFOV, val="depth",
                            saveto=dst_path, y_fudge=Y_FUDGE)
        # if len(points) == 0:
        #     print(img_file)
    #     to_pickle.append(points)
    # dst_path = os.path.join(output_path, *sinfo)
    # os.makedirs(dst_path, exist_ok=True)
    # pkl_path = os.path.join(dst_path, f'pcd-{sinfo[2]}.pkl')
    # pickle.dump(to_pickle, open(pkl_path, 'wb'))  
    # if len(to_pickle) < 5:
    #     logging.warning(f'{sinfo} has less than 5 valid data.')



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
    for sid in tqdm(sorted(os.listdir(input_path))):
        for seq in os.listdir(os.path.join(input_path,sid)):
            for view in os.listdir(os.path.join(input_path,sid,seq)):
                for img_path in os.listdir(os.path.join(input_path,sid,seq,view,'PCDs')):
                    img_groups[(sid, seq, view,'PCDs_offline_depths')].append(os.path.join(input_path,sid,seq,view, 'PCDs',img_path))
                    total_files += 1

    logging.info(f'Total files listed: {total_files}')

    progress = tqdm(total=len(img_groups), desc='Pretreating', unit='folder')

    with mp.Pool(workers) as pool:
        logging.info(f'Start pretreating {input_path}')
        for _ in pool.imap_unordered(partial(pcd2depth, output_path=output_path, img_size=img_size, verbose=verbose, dataset=dataset), img_groups.items()):
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
