import os
import sys
import shutil 
from tqdm import tqdm
from multiprocessing import Pool
from time import sleep
import argparse


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='/home1/data/OUMVLP_raw', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--output_path', default='/home1/data/OUMVLP_rearranged', type=str,
                    help='Root path for output.')


opt = parser.parse_args()

INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path


def mv_dir(src,dst):
    shutil.copytree(src, dst)
    print(src,dst)

sils_name_list = os.listdir(INPUT_PATH)
name_space = 'Silhouette_'
views = sorted(list(set([each.replace(name_space,'').split('-')[0] for each in sils_name_list])))
seqs = sorted(list(set([each.replace(name_space,'').split('-')[1] for each in sils_name_list])))
ids = list()
for each in sils_name_list:
    ids.extend(os.listdir(os.path.join(INPUT_PATH, each)))

# print(len(ids),len(set(ids)))
# print(views,seqs)

progress = tqdm(total=len(set(ids)))
# pool = Pool(8)

# print(sorted(set(ids)))


results = list()
pid = 0
for _id in sorted(set(ids)):
    progress.update(1)
    for _view in views:
        for _seq in seqs:
            seq_info = [_id, _seq ,_view]
            name = name_space + _view + '-' + _seq + '/' + _id 
            src = os.path.join(INPUT_PATH,name)
            dst = os.path.join(OUTPUT_PATH,*seq_info)
            if os.path.exists(src):
                try:
                    if os.path.exists(dst):
                        pass
                    else:
                        os.makedirs(dst)
                    for subfile in os.listdir(src):
                        os.symlink(os.path.join(src,subfile), os.path.join(dst,subfile))
                except OSError as err:
                    print(err)

                # print(src,dst,'success!')
