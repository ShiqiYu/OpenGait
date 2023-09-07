import pickle
from tqdm import tqdm
from pathlib import Path
import os
import os.path as osp
import argparse
import logging

'''
    gernerate the 17 Number of Pose Points Format from 18 Number of Pose Points
    OUMVLP 17
               # keypoints = {
            #     0: "nose",
            #     1: "left_eye",
            #     2: "right_eye",
            #     3: "left_ear",
            #     4: "right_ear",
            #     5: "left_shoulder",
            #     6: "right_shoulder",
            #     7: "left_elbow",
            #     8: "right_elbow",
            #     9: "left_wrist",
            #     10: "right_wrist",
            #     11: "left_hip",
            #     12: "right_hip",
            #     13: "left_knee",
            #     14: "right_knee",
            #     15: "left_ankle",
            #     16: "right_ankle"
            # }
    OUMVLP 18
    mask=[0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10]
           # keypoints = {
            #     0: "nose",  
            #     1: "neck",
            #     2: "Rshoulder",
            #     3: "Relbow",
            #     4: "Rwrist",
            #     5: "Lshoudler",
            #     6: "Lelbow",
            #     7: "Lwrist",
            #     8: "Rhip",
            #     9: "Rknee",
            #     10: "Rankle",
            #     11: "Lhip",
            #     12: "Lknee",
            #     13: "Lankle",
            #     14: "Reye",
            #     15: "Leye",
            #     16: "Rear",
            #     17: "Lear"
            # }
'''

def ToOUMVLP17(input_path: Path, output_path: Path):
    mask=[0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10]
    TOTAL_SUBJECTS = 10307
    progress = tqdm(total=TOTAL_SUBJECTS)

    for subject in input_path.iterdir():
        output_subject = subject.name
        for seq in subject.iterdir():
            output_seq = seq.name
            for view in seq.iterdir():
                src = os.path.join(view, f"{view.name}.pkl")
                dst = os.path.join(output_path, output_subject, output_seq, view.name)
                os.makedirs(dst, exist_ok=True)
                with open(src,'rb') as f:
                    srcdata = pickle.load(f)
                    #[T,18,3]
                data = srcdata[...,mask,:].copy()
                # #[T,17,3]
                pkl_path = os.path.join(dst,f'{view.name}.pkl')
                pickle.dump(data,open(pkl_path,'wb')) 
        progress.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGait dataset pretreatment module.')
    parser.add_argument('-i', '--input_path', default='', type=str, help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='', type=str, help='Output path of pickled dataset.')
    parser.add_argument('-l', '--log_to_file', default='./pretreatment.log', type=str, help='Log file path. Default: ./pretreatment.log')
    args = parser.parse_args()
    logging.info('Begin')
    ToOUMVLP17(input_path=Path(args.input_path), output_path=Path(args.output_path))
    logging.info('Done')
