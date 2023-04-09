import argparse
import os
from pathlib import Path
import tqdm
import cv2
import tarfile
import zipfile
from functools import partial
import numpy as np
import pickle
import multiprocessing as mp


def make_pkl_for_one_person(id_, output_path, img_size=64):
    if id_.split(".")[-1] != "tar" or not os.path.exists(os.path.join(output_path, id_)):
        return
    with tarfile.TarFile(os.path.join(output_path, id_)) as f:
        f.extractall(output_path)
    os.remove(os.path.join(output_path, id_))
    id_path = id_.split(".")[0]
    input_path = os.path.join(output_path, "forTrain", id_path)
    base_pkl_path = os.path.join(output_path, "opengait", id_path)
    if not os.path.isdir(input_path):
        print("Path not found: "+input_path)
    for height in sorted(os.listdir(input_path)):
        height_path = os.path.join(input_path, height)
        for scene in sorted(os.listdir(height_path)):
            scene_path = os.path.join(height_path, scene)
            for type_ in sorted(os.listdir(scene_path)):
                type_path = os.path.join(scene_path, type_)
                for view in sorted(os.listdir(type_path)):
                    view_path = os.path.join(type_path, view)
                    for num in sorted(os.listdir(view_path)):
                        num_path = os.path.join(view_path, num)
                        imgs = []
                        for file_ in sorted(os.listdir(num_path)):
                            img = cv2.imread(os.path.join(
                                num_path, file_), cv2.IMREAD_GRAYSCALE)
                            if img_size != img.shape[0]:
                                img = cv2.resize(
                                    img, dsize=(img_size, img_size))
                            imgs.append(img)
                        if len(imgs) > 5:
                            pkl_path = os.path.join(
                                base_pkl_path, f"{height}-{scene}-{type_}-{num}", view)
                            os.makedirs(pkl_path, exist_ok=True)
                            pickle.dump(np.asarray(imgs), open(
                                os.path.join(pkl_path, f"{view}.pkl"), "wb"))
                        else:
                            print("No enough imgs: "+num_path)


def extractall(base_path: Path, output_path: Path, workers=1, img_size=64) -> None:
    """Extract all archives in base_path to output_path.

    Args:
        base_path (Path): Path to the directory containing the archives.
        output_path (Path): Path to the directory to extract the archives to.
    """

    os.makedirs(output_path, exist_ok=True)
    print("Unzipping train set...")
    with open(os.path.join(base_path, 'train001-500.zip'), 'rb') as f:
        z = zipfile.ZipFile(f)
        z.extractall(output_path)
    print("Unzipping validation set...")
    with open(os.path.join(base_path, 'val501-614.zip'), 'rb') as f:
        z = zipfile.ZipFile(f)
        z.extractall(output_path)
    print("Unzipping test set...")
    with open(os.path.join(base_path, 'test615-1014.zip'), 'rb') as f:
        z = zipfile.ZipFile(f)
        z.extractall(output_path)
    print("Extracting tar file...")
    os.makedirs(os.path.join(output_path,"forTrain"))
    os.makedirs(os.path.join(output_path,"opengait"))
    ids = os.listdir(os.path.join(output_path))
    progress = tqdm.tqdm(total=len(ids), desc='Pretreating', unit='id')

    with mp.Pool(workers) as pool:
        for _ in pool.imap_unordered(partial(make_pkl_for_one_person, output_path=output_path, img_size=img_size), ids):
            progress.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CASIA-E extractor')
    parser.add_argument('-b', '--input_path', type=str,
                        required=True, help='Base path to CASIA-E zip files')
    parser.add_argument('-o', '--output_path', type=str,
                        required=True, help='Output path for extracted files. The pickle files are generated in ${output_path}/opengait/')
    parser.add_argument('-s', '--img_size', default=64,
                        type=int, help='Image resizing size. Default 64')
    parser.add_argument('-n', '--num_workers',
                        type=int, default=1, help='Number of workers')
    args = parser.parse_args()

    extractall(Path(args.input_path), Path(args.output_path),
               args.num_workers, args.img_size)
