import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm



def rearrange(input_path: Path) -> None:
    for id in tqdm(os.listdir(input_path)):
            for device in os.listdir(os.path.join(input_path,id)):
                    for seq in os.listdir(os.path.join(input_path,id,device)):
                            for pkl in ['image', 'lidar', 'range_pkl', 'smpl', 'kp3d']:
                                    pkl_dir = os.path.join(input_path, id, device, seq, pkl, pkl+'.pkl')
                                    target_dir = os.path.join(input_path, id, device, seq, pkl+'.pkl')
                                    shutil.move(pkl_dir, target_dir)
                                    os.rmdir(os.path.join(input_path, id, device, seq, pkl))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FreeGait rearrange tool')
    parser.add_argument('-i', '--input_path', required=True, type=str,
                        help='Root path of raw dataset.')

    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    rearrange(input_path)
    print('Done!')
