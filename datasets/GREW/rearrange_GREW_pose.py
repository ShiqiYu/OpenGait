import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm

TOTAL_Test = 24000
TOTAL_Train = 20000

def rearrange_train(train_path: Path, output_path: Path) -> None:
    progress = tqdm(total=TOTAL_Train)
    for sid in train_path.iterdir():
        if not sid.is_dir():
            continue
        for sub_seq in sid.iterdir():
            if not sub_seq.is_dir():
                continue
            for subfile in os.listdir(sub_seq):
                src = os.path.join(train_path, sid.name, sub_seq.name)
                dst = os.path.join(output_path, sid.name+'train', '00', sub_seq.name)
                os.makedirs(dst,exist_ok=True)
                if subfile not in os.listdir(dst) and subfile.endswith('_2d_pose.txt'):
                    pose_subfile = 'pose_'+subfile
                    os.symlink(os.path.join(src, subfile),
                               os.path.join(dst, pose_subfile))
        progress.update(1)

def rearrange_test(test_path: Path, output_path: Path) -> None:
    # for gallery
    gallery = Path(os.path.join(test_path, 'gallery'))
    probe = Path(os.path.join(test_path, 'probe'))
    progress = tqdm(total=TOTAL_Test)
    for sid in gallery.iterdir():
        if not sid.is_dir():
            continue
        cnt = 1
        for sub_seq in sid.iterdir():
            if not sub_seq.is_dir():
                continue
            for subfile in sorted(os.listdir(sub_seq)):
                src = os.path.join(gallery, sid.name, sub_seq.name)
                dst = os.path.join(output_path, sid.name, '%02d'%cnt, sub_seq.name)
                os.makedirs(dst,exist_ok=True)
                if subfile not in os.listdir(dst) and subfile.endswith('_2d_pose.txt'):
                    pose_subfile = 'pose_'+subfile
                    os.symlink(os.path.join(src, subfile),
                               os.path.join(dst, pose_subfile))
            cnt += 1
            progress.update(1)
    # for probe
    for sub_seq in probe.iterdir():
        if not sub_seq.is_dir():
            continue
        for subfile in os.listdir(sub_seq):
            src = os.path.join(probe, sub_seq.name)
            dst = os.path.join(output_path, 'probe', '03', sub_seq.name)
            os.makedirs(dst,exist_ok=True)
            if subfile not in os.listdir(dst) and subfile.endswith('_2d_pose.txt'):
                pose_subfile = 'pose_'+subfile
                os.symlink(os.path.join(src, subfile),
                            os.path.join(dst, pose_subfile))
            progress.update(1)

def rearrange_GREW(input_path: Path, output_path: Path) -> None:
    os.makedirs(output_path, exist_ok=True)

    for folder in input_path.iterdir():
        if not folder.is_dir():
            continue

        print(f'Rearranging {folder}')
        if folder.name == 'train':
            rearrange_train(folder,output_path)
        if folder.name == 'test':
            rearrange_test(folder, output_path)
        if folder.name == 'distractor':
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GREW rearrange tool')
    parser.add_argument('-i', '--input_path', required=True, type=str,
                        help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='GREW_rearranged', type=str,
                        help='Root path for output.')

    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    rearrange_GREW(input_path, output_path)