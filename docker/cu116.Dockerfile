FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# Set environment variables
ENV NVENCODE_CFLAGS="-I/usr/local/cuda/include"
ENV CV_VERSION=4.2.0
ENV DEBIAN_FRONTEND=noninteractive

# Get all dependencies
RUN apt-get update && apt-get install -y \
    git zip unzip libssl-dev libcairo2-dev lsb-release libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev software-properties-common \
    build-essential cmake pkg-config libapr1-dev autoconf automake libtool curl libc6 libboost-all-dev debconf libomp5 libstdc++6 \
    libqt5core5a libqt5xml5 libqt5gui5 libqt5widgets5 libqt5concurrent5 libqt5opengl5 libcap2 libusb-1.0-0 libatk-adaptor neovim \
    python3-pip python3-tornado python3-dev python3-numpy python3-virtualenv libpcl-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev \
    libsuitesparse-dev python3-pcl pcl-tools libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
    libpng-dev libtiff-dev libdc1394-22-dev xfce4-terminal &&\
    rm -rf /var/lib/apt/lists/*

# Set python3 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# OpenCV with CUDA support
## In case this is needed, uncomment the following lines and comment the pip3 install opencv-python line
### -------------------------------------------------------------------------------
# WORKDIR /opencv
# RUN git clone https://github.com/opencv/opencv.git -b $CV_VERSION &&\
#     git clone https://github.com/opencv/opencv_contrib.git -b $CV_VERSION

# RUN mkdir opencvfix && cd opencvfix &&\
#     git clone https://github.com/opencv/opencv.git -b 4.5.2 &&\
#     cd opencv/cmake &&\
#     cp -r FindCUDA /opencv/opencv/cmake/ &&\
#     cp FindCUDA.cmake /opencv/opencv/cmake/ &&\
#     cp FindCUDNN.cmake /opencv/opencv/cmake/ &&\
#     cp OpenCVDetectCUDA.cmake /opencv/opencv/cmake/
 
# WORKDIR /opencv/opencv/build

# RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
# -D CMAKE_INSTALL_PREFIX=/usr/local \
# -D OPENCV_GENERATE_PKGCONFIG=ON \
# -D BUILD_EXAMPLES=OFF \
# -D INSTALL_PYTHON_EXAMPLES=OFF \
# -D INSTALL_C_EXAMPLES=OFF \
# -D PYTHON_EXECUTABLE=$(which python2) \
# -D PYTHON3_EXECUTABLE=$(which python3) \
# -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
# -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
# -D BUILD_opencv_python2=ON \
# -D BUILD_opencv_python3=ON \
# -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
# -D WITH_GSTREAMER=ON \
# -D WITH_CUDA=ON \
# -D ENABLE_PRECOMPILED_HEADERS=OFF \
# .. &&\
# make -j$(nproc) &&\
# make install &&\
# ldconfig &&\
# rm -rf /opencv

# WORKDIR /
# ENV OpenCV_DIR=/usr/share/OpenCV
### -------------------------------------------------------------------------------

RUN pip3 install opencv-python
RUN pip3 install pyyaml
RUN pip3 install tensorboard
RUN pip3 install tqdm
RUN pip3 install py7zr
RUN pip3 install kornia
RUN pip3 install einops
# Install PyTorch and torchvision with CUDA support
RUN pip3 install torch=="1.13.1+cu116" torchvision=="0.14.1+cu116" --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install scikit-learn
RUN pip3 install imageio
RUN pip3 install matplotlib
# Set the environment variables
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

WORKDIR /opengait

ENV NVIDIA_VISIBLE_DEVICES="all" \
    OpenCV_DIR=/usr/share/OpenCV \
    NVIDIA_DRIVER_CAPABILITIES="video,compute,utility,graphics" \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib \
    QT_GRAPHICSSYSTEM="native"

