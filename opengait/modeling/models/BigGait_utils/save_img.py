from os import path as osp
import os
import pickle
from PIL import Image
import imageio
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import cv2


def pca_image(data, mask, root, model_name, dataset, n_components=3, is_return=False):
    features = data['embeddings']
    ns,hw,c = features.shape
    features = features.reshape(ns*hw,c)
    mask = mask.reshape(ns*hw)

    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features[mask != 0])
    pca_features = minmax_scale(pca_features, (0,255), axis=1)
    # pca_features = minmax_scale(pca_features, (0,255), axis=0)

    norm_features = np.zeros_like(mask,dtype=np.uint8).reshape(ns*hw,1).repeat(n_components,axis=1)
    norm_features[mask != 0] = pca_features

    if is_return:
        norm_features = norm_features.reshape(1,ns,64,32,n_components)[...,:3].transpose(0,1,4,2,3) # 
        return norm_features
    
    s = 20
    assert ns % s == 0
    norm_features = norm_features.reshape(ns//s,s,64,32,n_components)[...,:3].transpose(0,1,4,2,3)
    data['embeddings'] = norm_features
    save_image(data, root, model_name, dataset, need='image')



def save_image(data, root, model_name, dataset, need='image', mask=None):
    images, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views'] # n s c h w
    if "image" in need:
        root_path = os.path.join(root, dataset, model_name+'_image')
        os.makedirs(os.path.join(root_path),exist_ok=True)
        for i, id in enumerate(label[:]):
            tmp = os.path.join(root_path, str(id).zfill(5), str(seq_type[i]), str(view[i]))
            os.makedirs(tmp, exist_ok=True)
            mb = None if mask is None else mask[i]
            save_func(tmp, images[i], need, mb)
            save_gif(tmp, tmp, str(view[i]))

    if 'pkl' in need:
        root_path = os.path.join(root, dataset, model_name+'_pkl')
        os.makedirs(os.path.join(root_path),exist_ok=True)
        for i, id in enumerate(label[:]):        
            tmp = os.path.join(root_path, str(id).zfill(5), str(seq_type[i]), str(view[i]))
            os.makedirs(tmp, exist_ok=True)
            mb = None if mask is None else mask[i]
            save_func(tmp, images[i], 'pkl', mb)

    if 'w' in need:
        root_path = os.path.join(root, dataset, model_name+'_w')
        os.makedirs(os.path.join(root_path),exist_ok=True)
        for i, id in enumerate(label[:]):        
            tmp = os.path.join(root_path, str(id).zfill(5), str(seq_type[i]), str(view[i]))
            os.makedirs(tmp, exist_ok=True)
            mb = None if mask is None else mask[i]
            save_func(tmp, data['w'], 'w', mb)
    return

def save_func(tmp, data, ipts_type='image', mask=None):
    if 'image' in ipts_type :
        for i, con in enumerate(data):
            if con.shape[0] == 1:
                if 'jet' in ipts_type :
                    im = ((cv2.applyColorMap(con[0], cv2.COLORMAP_JET) * 0.5)[...,::-1] + 1.0*mask[i])
                    # im = mask[i]
                    im = np.clip(im,0,255).astype(np.uint8)
                    im = Image.fromarray(im, mode='RGB') # [h,w,c]
                else:
                    im = Image.fromarray(con[0], mode='L')
            else:
                im = Image.fromarray(con.transpose(1,2,0), mode='RGB')
            im.save(os.path.join(tmp, '%03d.png' % i))
    elif ipts_type == 'pkl':
        with open(os.path.join(tmp,'00.pkl'), 'wb') as f:
            pickle.dump(data[:,0,:,:], f)
    elif ipts_type == 'w':
        for i in range(len(data)):
            with open(os.path.join(tmp, str(i).zfill(2) + '.pkl'), 'wb') as f:
                pickle.dump(data[i], f)

def save_gif(image_folder, save_folder, name="movie"):
    images = []
    filenames = sorted(glob(osp.join(image_folder, '*.png')))
    # print(filenames)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(save_folder, f'{name}.gif'), images, duration=50, loop=0)
