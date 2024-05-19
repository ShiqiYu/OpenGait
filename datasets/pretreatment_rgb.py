import os
from time import time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

SRC_0 = 'rgb_input_path'
DST_0 = 'rgb_pkl_output_path'

SRC = SRC_0             # Path_of_RGB_rearranged
DST = DST_0             # Path_of_RGB_256128pkl_PadResized

def resize_with_padding(img, target_size):
    h, w, _ = img.shape
    target_h, target_w = target_size
    resized_img = cv2.resize(img, (int(w * target_h / h), target_h))
    padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - resized_img.shape[1]) // 2
    if x_offset < 0 :
        x_offset = abs(x_offset)
        padded_img = resized_img[:, x_offset:x_offset+target_w,:]
    else: 
        padded_img[:, x_offset:x_offset + resized_img.shape[1]] = resized_img
    return padded_img

def job(src, id):
    for ty in sorted(os.listdir(os.path.join(src, id))):
        for vi in sorted(os.listdir(os.path.join(src, id, ty))):
            exist_file = os.path.join(DST, id, ty, vi, vi+"-aligned-rgbs.pkl")
            if os.path.exists(exist_file):
                print('Have Passed: ' + DST + '/' + id + '/' + ty)
                continue
            ratios = []
            aligned_imgs = []
            for img_file in sorted(os.listdir(os.path.join(src, id, ty, vi))):
                img_path = os.path.join(src, id, ty, vi, img_file)
                img = cv2.imread(img_path)
                ratio = img.shape[1]/img.shape[0]
                ratios.append(ratio)
                aligned_img = np.transpose(cv2.cvtColor(resize_with_padding(img, (256, 128)), cv2.COLOR_BGR2RGB), (2, 0, 1))
                aligned_imgs.append(aligned_img)
            if len(aligned_imgs) > 0:
                output_path = os.path.join(DST, id, ty, vi)
                os.makedirs(output_path, exist_ok=True)
                pickle.dump(np.asarray(aligned_imgs), open(os.path.join(output_path, vi+"-aligned-rgbs.pkl"), "wb"))
                pickle.dump(np.asarray(ratios), open(os.path.join(output_path, vi+"-ratios.pkl"), "wb"))
            print('Successfully saved: ' + DST + '/' + id + '/' + ty +  '/' + vi)
                    
if __name__ == '__main__':
    a = time()
    po = Pool(8)
    src_path = SRC
    
    cnt = 0
    need_data = sorted(os.listdir(src_path))
    for id in tqdm(need_data[:]):
        po.apply_async(job,(src_path, id,))
        cnt = cnt + 1
    
    print('---START---')
    po.close()
    po.join()
    print(cnt)

    t = time() - a
    print('---END---{}'.format(t))
