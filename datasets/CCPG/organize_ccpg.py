import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import argparse


T_W = 64
T_H = 64


def cut_img(img):
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_AREA)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CCPG dataset Preprocessing.')
    parser.add_argument('--sil_path', default='', type=str,
                        help='Root path of raw silhouette dataset.')
    parser.add_argument('--rgb_path', default='', type=str,
                        help='Root path of raw RGB dataset.')
    parser.add_argument('-o', '--output_path', default='',
                        type=str, help='Output path of pickled dataset.')
    args = parser.parse_args()

    RGB_SIZE = (128, 128)
    for _id in tqdm(sorted(os.listdir(args.sil_path))):
        for _type in sorted(os.listdir(os.path.join(args.rgb_path, _id))):
            for _view in sorted(os.listdir(os.path.join(args.rgb_path, _id, _type))):
                imgs = []
                segs = []
                ratios = []
                aligned_segs = []
                for img_file in sorted(os.listdir(os.path.join(args.rgb_path, _id, _type, _view))):
                    seg_file = img_file.split(".")[0]+".png"
                    img_path = os.path.join(
                        args.rgb_path, _id, _type, _view, img_file)
                    seg_path = os.path.join(
                        args.rgb_path, _id, _type, _view, seg_file)
                    if not os.path.exists(seg_path):
                        print("Not Found: "+seg_path)
                        continue
                    img = cv2.imread(img_path)
                    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                    ratio = img.shape[1]/img.shape[0]
                    aligned_seg = cut_img(seg)
                    img = np.transpose(cv2.cvtColor(cv2.resize(
                        img, RGB_SIZE), cv2.COLOR_BGR2RGB), (2, 0, 1))
                    imgs.append(img)
                    segs.append(cv2.resize(
                        seg, RGB_SIZE))
                    aligned_segs.append(aligned_seg)
                    ratios.append(ratio)
                if len(imgs) > 0:
                    output_path = os.path.join(
                        args.output_path, _id, _type, _view)
                    os.makedirs(output_path, exist_ok=True)
                    pickle.dump(np.asarray(imgs), open(os.path.join(
                        output_path, _view+"-rgbs.pkl"), "wb"))
                    pickle.dump(np.asarray(segs), open(os.path.join(
                        output_path, _view+"-sils.pkl"), "wb"))
                    pickle.dump(np.asarray(ratios), open(os.path.join(
                        output_path, _view+"-ratios.pkl"), "wb"))
                    pickle.dump(np.asarray(aligned_segs), open(os.path.join(
                        output_path, _view+"-aligned-sils.pkl"), "wb"))
                else:
                    print("No imgs Found: " +
                          os.path.join(args.rgb_path, _id, _type, _view))
                    continue
