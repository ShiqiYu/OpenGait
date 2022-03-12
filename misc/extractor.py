import argparse
import os
from pathlib import Path

import py7zr
from tqdm import tqdm


def extractall(base_path: Path, output_path: Path, passwords) -> None:
    """Extract all archives in base_path to output_path.

    Args:
        base_path (Path): Path to the directory containing the archives.
        output_path (Path): Path to the directory to extract the archives to.
    """

    os.makedirs(output_path, exist_ok=True)
    for file_path in tqdm(list(base_path.rglob('Silhouette_*.7z'))):
        if output_path.joinpath(file_path.stem).exists():
            continue
        with py7zr.SevenZipFile(file_path, password=passwords) as archive:
            total_items = len(
                [f for f in archive.getnames() if f.endswith('.png')]
            )
            archive.extractall(output_path)

        extracted_files = len(
            list(output_path.joinpath(file_path.stem).rglob('*.png')))

        assert extracted_files == total_items, f'{extracted_files} != {total_items}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OUMVLP extractor')
    parser.add_argument('-b', '--input_path', type=str,
                        required=True, help='Base path to OUMVLP .7z files')
    parser.add_argument('-o', '--output_path', type=str,
                        required=True, help='Output path for extracted files')
    parser.add_argument('-p', '--password', type=str,
                        required=True, help='password for extracted files')

    args = parser.parse_args()

    extractall(Path(args.input_path), Path(args.output_path), args.password)
