# Modified from https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.3/paddleseg/utils/download.py

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import requests
import time
import sys
import zipfile
lasttime = time.time()
FLUSH_INTERVAL = 0.1


def progress(str, end=False):
    global lasttime
    if end:
        str += "\n"
        lasttime = 0
    if time.time() - lasttime >= FLUSH_INTERVAL:
        sys.stdout.write("\r%s" % str)
        lasttime = time.time()
        sys.stdout.flush()


def _download_file(url, savepath, print_progress):
    if print_progress:
        print("Connecting to {}".format(url))
    r = requests.get(url, stream=True, timeout=15)
    total_length = r.headers.get('content-length')

    if total_length is None:
        with open(savepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    else:
        with open(savepath, 'wb') as f:
            dl = 0
            total_length = int(total_length)
            if print_progress:
                print("Downloading %s" % os.path.basename(savepath))
            for data in r.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                if print_progress:
                    done = int(50 * dl / total_length)
                    progress("[%-50s] %.2f%%" %
                             ('=' * done, float(100 * dl) / total_length))
        if print_progress:
            progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)


def _uncompress_file_zip(filepath, extrapath):
    files = zipfile.ZipFile(filepath, 'r')
    filelist = files.namelist()
    rootpath = filelist[0]
    total_num = len(filelist)
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath


def download_file_and_uncompress(url,
                                 savepath=None,
                                 print_progress=True,
                                 replace=False,
                                 extrapath=None,
                                 delete_file=True):
    if savepath is None:
        savepath = "."
    if extrapath is None:
        extrapath = "."
    savename = url.split("/")[-1]
    if not savename.endswith("zip"):
        raise NotImplementedError(
            "Only support zip file, but got {}!".format(savename))
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    savepath = os.path.join(savepath, savename)
    savename = ".".join(savename.split(".")[:-1])

    if replace:
        if os.path.exists(savepath):
            shutil.rmtree(savepath)

    if not os.path.exists(savename):
        if not os.path.exists(savepath):
            _download_file(url, savepath, print_progress)

        if print_progress:
            print("Uncompress %s" % os.path.basename(savepath))
        for total_num, index, rootpath in _uncompress_file_zip(savepath, extrapath):
            if print_progress:
                done = int(50 * float(index) / total_num)
                progress(
                    "[%-50s] %.2f%%" % ('=' * done, float(100 * index) / total_num))
        if print_progress:
            progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)

        if delete_file:
            os.remove(savepath)

    return rootpath


if __name__ == "__main__":
    url = "https://github.com/ShiqiYu/OpenGait/releases/download/v1.0-beta/pretrained_model.zip"
    download_file_and_uncompress(
        url=url, extrapath='output')

    print("Pretrained model download success!")
