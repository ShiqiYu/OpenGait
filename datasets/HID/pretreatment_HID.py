import os
import cv2
import numpy as np
import argparse
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_train_path', default='', type=str,
                    help='Root path of train.')
parser.add_argument('--input_gallery_path', default='', type=str,
                    help='Root path of gallery.')
parser.add_argument('--input_probe_path', default='', type=str,
                    help='Root path of probe.')
parser.add_argument('--output_path', default='', type=str,
                    help='Root path for output.')

opt = parser.parse_args()

OUTPUT_PATH = opt.output_path
print('Pretreatment Start.\n'
      'Input train path: {}\n'
      'Input gallery path: {}\n'
      'Input probe path: {}\n'
      'Output path: {}\n'.format(
          opt.input_train_path, opt.input_gallery_path, opt.input_probe_path, OUTPUT_PATH))

INPUT_PATH = opt.input_train_path
print("Walk the input train path")
id_list = os.listdir(INPUT_PATH)
id_list.sort()

for _id in tqdm(id_list):
    seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
    seq_type.sort()
    for _seq_type in seq_type:
        out_dir = os.path.join(OUTPUT_PATH, _id, _seq_type, "default")
        count_frame = 0
        all_imgs = []
        frame_list = sorted(os.listdir(
            os.path.join(INPUT_PATH, _id, _seq_type)))
        for _frame_name in frame_list:
            frame_path = os.path.join(
                INPUT_PATH, _id, _seq_type, _frame_name)
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Save the img
                all_imgs.append(img)
                count_frame += 1

        all_imgs = np.asarray(all_imgs)

        if count_frame > 0:
            os.makedirs(out_dir, exist_ok=True)
            all_imgs_pkl = os.path.join(out_dir, '{}.pkl'.format(_seq_type))
            pickle.dump(all_imgs, open(all_imgs_pkl, 'wb'))

        # Warn if the sequence contains less than 5 frames
        if count_frame < 5:
            print('Seq:{}-{}, less than 5 valid data.'.format(_id, _seq_type))

print("Walk the input gallery path")
INPUT_PATH = opt.input_gallery_path
id_list = os.listdir(INPUT_PATH)
id_list.sort()
for _id in tqdm(id_list):
    seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
    seq_type.sort()
    for _seq_type in seq_type:
        out_dir = os.path.join(OUTPUT_PATH, _id, _seq_type, "default")
        count_frame = 0
        all_imgs = []
        frame_list = sorted(os.listdir(
            os.path.join(INPUT_PATH, _id, _seq_type)))
        for _frame_name in frame_list:
            frame_path = os.path.join(
                INPUT_PATH, _id, _seq_type, _frame_name)
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Save the img
                all_imgs.append(img)
                count_frame += 1

        all_imgs = np.asarray(all_imgs)

        if count_frame > 0:
            os.makedirs(out_dir, exist_ok=True)
            all_imgs_pkl = os.path.join(out_dir, '{}.pkl'.format(_seq_type))
            pickle.dump(all_imgs, open(all_imgs_pkl, 'wb'))

        # Warn if the sequence contains less than 5 frames
        if count_frame < 5:
            print('Seq:{}-{}, less than 5 valid data.'.format(_id, _seq_type))
    print("Finish {}".format(_id))

print("Walk the input probe path")
INPUT_PATH = opt.input_probe_path
seq_type = os.listdir(INPUT_PATH)
seq_type.sort()

_id = "probe"
for _seq_type in tqdm(seq_type):
    out_dir = os.path.join(OUTPUT_PATH, _id, _seq_type, "default")
    count_frame = 0
    all_imgs = []
    frame_list = sorted(os.listdir(
        os.path.join(INPUT_PATH, _seq_type)))
    for _frame_name in frame_list:
        frame_path = os.path.join(
            INPUT_PATH, _seq_type, _frame_name)
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Save the img
            all_imgs.append(img)
            count_frame += 1
    all_imgs = np.asarray(all_imgs)
    if count_frame > 0:
        os.makedirs(out_dir, exist_ok=True)
        all_imgs_pkl = os.path.join(out_dir, '{}.pkl'.format(_seq_type))
        pickle.dump(all_imgs, open(all_imgs_pkl, 'wb'))
    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        print('Seq:{}-{}, less than 5 valid data.'.format(_id, _seq_type))
