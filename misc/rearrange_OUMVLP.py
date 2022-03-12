import os
import shutil
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='OUMVLP', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--output_path', default='OUMVLP_rearranged', type=str,
                    help='Root path for output.')


opt = parser.parse_args()

INPUT_PATH = os.path.abspath(opt.input_path)
OUTPUT_PATH = os.path.abspath(opt.output_path)


sils_name_list = os.listdir(INPUT_PATH)
name_space = 'Silhouette_'
views = sorted(list(
    set([each.replace(name_space, '').split('-')[0] for each in sils_name_list])))
seqs = sorted(list(
    set([each.replace(name_space, '').split('-')[1] for each in sils_name_list])))
ids = list()
for each in sils_name_list:
    ids.extend(os.listdir(os.path.join(INPUT_PATH, each)))

progress = tqdm(total=len(set(ids)))

results = list()
for _id in sorted(set(ids)):
    progress.update(1)
    for _view in views:
        for _seq in seqs:
            seq_info = [_id, _seq, _view]
            name = name_space + _view + '-' + _seq + '/' + _id
            src = os.path.join(INPUT_PATH, name)
            dst = os.path.join(OUTPUT_PATH, *seq_info)
            if os.path.exists(src):
                try:
                    os.makedirs(dst, exist_ok=True)
                    for subfile in os.listdir(src):
                        if subfile not in os.listdir(dst):  # subfile exits, pass
                            os.symlink(os.path.join(src, subfile),
                                       os.path.join(dst, subfile))
                except OSError as err:
                    print(err)