import os
import argparse
from pathlib import Path
import shutil


def merge(sils_path, smpls_path, output_path, link):
    if link == 'hard':
        link_method = os.link
    elif link == 'soft':
        link_method = os.symlink
    else:
        link_method = shutil.copyfile
    for _id in os.listdir(sils_path):
        id_path = os.path.join(sils_path, _id)
        for _type in os.listdir(id_path):
            type_path = os.path.join(id_path, _type)
            for _view in os.listdir(type_path):
                view_path = os.path.join(type_path, _view)
                for _seq in os.listdir(view_path):
                    sils_seq_path = os.path.join(view_path, _seq)
                    smpls_seq_path = os.path.join(
                        smpls_path, _id, _type, _view, _seq)
                    output_seq_path = os.path.join(output_path, _id, _type, _view)
                    os.makedirs(output_seq_path, exist_ok=True)
                    link_method(sils_seq_path, os.path.join(
                        output_seq_path, "sils-"+_seq))
                    link_method(smpls_seq_path, os.path.join(
                        output_seq_path, "smpls-"+_seq))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gait3D dataset mergence.')
    parser.add_argument('--sils_path', default='', type=str,
                        help='Root path of raw silhs dataset.')
    parser.add_argument('--smpls_path', default='', type=str,
                        help='Root path of raw smpls dataset.')
    parser.add_argument('-o', '--output_path', default='',
                        type=str, help='Output path of pickled dataset.')
    parser.add_argument('-l', '--link', default='hard', type=str,
                        choices=['hard', 'soft', 'copy'], help='Link type of output data.')
    args = parser.parse_args()

    merge(sils_path=Path(args.sils_path), smpls_path=Path(
        args.smpls_path), output_path=Path(args.output_path), link=args.link)
    