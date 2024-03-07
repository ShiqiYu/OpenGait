import os
import sys
import argparse
from tqdm import tqdm
from glob import glob

def get_args():
    parser = argparse.ArgumentParser(description='Symlink silouette data and pose data into the same folder for SkeletonGait++ training.')
    parser.add_argument('--heatmap_data_path', type=str, required=True, help="path of heatmap data, must be the absolute path.")
    parser.add_argument('--silhouette_data_path', type=str, required=True, help="path of silouette data, must be the absolute path.")
    parser.add_argument('--dataset_pkl_ext_name', type=str, default='.pkl', help="The extent name for .pkl files of silouettes data.")
    parser.add_argument('--output_path', type=str, required=True, help="path of output data")
    opt = parser.parse_args()
    return opt

def main():
    opt = get_args()
    heatmap_data_path = opt.heatmap_data_path
    silhouette_data_path = opt.silhouette_data_path
    if not os.path.exists(heatmap_data_path):
        print(f"heatmap data path {heatmap_data_path} does not exist.")
        sys.exit(1)
    if not os.path.exists(silhouette_data_path):
        print(f"silouette data path {silhouette_data_path} does not exist.")
        sys.exit(1)
    
    all_heatmap_files = sorted(glob(os.path.join(heatmap_data_path, "*/*/*/*.pkl")))
    all_silouette_files = sorted(glob(os.path.join(silhouette_data_path, f"*/*/*/*{opt.dataset_pkl_ext_name}")))
    # print(len(all_heatmap_files), len(all_silouette_files))
    # assert len(all_heatmap_files) == len(all_silouette_files), "The number of heatmap files and silouette files are not equal."
    
    if len(all_heatmap_files) >= len(all_silouette_files):
        for heatmap_file in tqdm(all_heatmap_files):
            tmp_list = heatmap_file.split('/')
            sil_folder = os.path.join(silhouette_data_path, *tmp_list[-4:-1])
            if not os.path.exists(sil_folder):
                print(f"silouette folder {sil_folder} does not exist.")
                continue
            else:
                silouette_file = sorted(glob(os.path.join(sil_folder, f"*{opt.dataset_pkl_ext_name}")))[0]

            output_file = os.path.join(opt.output_path, *tmp_list[-4:-1])
            os.makedirs(output_file, exist_ok=True)
            os.system(f"ln -s {silouette_file} {output_file}/1_sil.pkl")
            os.system(f"ln -s {heatmap_file} {output_file}/0_heatmap.pkl")
    else:
        for silouette_file in tqdm(all_silouette_files):
            tmp_list = silouette_file.split('/')
            heatmap_folder = os.path.join(heatmap_data_path, *tmp_list[-4:-1])
            if not os.path.exists(heatmap_folder):
                print(f"heatmap folder {heatmap_folder} does not exist.")
                continue
            else:
                heatmap_file = sorted(glob(os.path.join(heatmap_folder, "*.pkl")))[0]

            output_file = os.path.join(opt.output_path, *tmp_list[-4:-1])
            os.makedirs(output_file, exist_ok=True)
            os.system(f"ln -s {silouette_file} {output_file}/1_sil.pkl")
            os.system(f"ln -s {heatmap_file} {output_file}/0_heatmap.pkl")

    print("Done! Output data is in ", opt.output_path)

    # for tmp_file in tqdm(iter_files):
    #     heatmap_file = all_heatmap_files[i]
    #     silouette_file = all_silouette_files[i]
    #     sil_tmp_list = silouette_file.split('/')
    #     heatmap_tmp_list = heatmap_file.split('/')
    #     if 

    #     output_file = os.path.join(opt.output_path, *tmp_list[-4:-1])
    #     os.makedirs(output_file, exist_ok=True)

    #     os.system(f"ln -s {silouette_file} {output_file}/1_sil.pkl")
    #     os.system(f"ln -s {heatmap_file} {output_file}/0_heatmap.pkl")

if __name__ == "__main__":
    main()
