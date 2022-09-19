import argparse
import os
import shutil
from pathlib import Path
from typing import Tuple

from tqdm import tqdm


TOTAL_SUBJECTS = 10307


def sanitize(name: str) -> Tuple[str, str]:
    return name.split('_')[1].split('-')


def rearrange(input_path: Path, output_path: Path) -> None:
    os.makedirs(output_path, exist_ok=True)

    for folder in input_path.iterdir():
        print(f'Rearranging {folder}')
        view, seq = sanitize(folder.name)
        progress = tqdm(total=TOTAL_SUBJECTS)
        for sid in folder.iterdir():
            src = os.path.join(input_path, f'Silhouette_{view}-{seq}', sid.name)
            dst = os.path.join(output_path, sid.name, seq, view)
            os.makedirs(dst, exist_ok=True)
            for subfile in os.listdir(src):
                if subfile not in os.listdir(dst) and subfile.endswith('.png'):
                    os.symlink(os.path.join(src, subfile),
                               os.path.join(dst, subfile))
                # else:
                #     os.remove(os.path.join(src, subfile))
            progress.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OUMVLP rearrange tool')
    parser.add_argument('-i', '--input_path', required=True, type=str,
                        help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='OUMVLP_rearranged', type=str,
                        help='Root path for output.')

    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    rearrange(input_path, output_path)
