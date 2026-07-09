FROM python:3.8-slim

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの作成
WORKDIR /app

# OpenGaitのクローン
RUN git clone https://github.com/ShiqiYu/OpenGait.git

# ホストの requirements.txt を OpenGait にコピー
COPY requirements.txt /app/OpenGait/

# 作業ディレクトリをOpenGaitに変更
WORKDIR /app/OpenGait

# Pythonパッケージのインストール
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade pip
FROM python:3.8-slim

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの作成
WORKDIR /app

# OpenGait のクローン
RUN git clone https://github.com/ShiqiYu/OpenGait.git

# requirements.txt を OpenGait にコピー
COPY requirements.txt /app/OpenGait/

# 作業ディレクトリを OpenGait に移動
WORKDIR /app/OpenGait

# Python パッケージのインストール
RUN pip install --upgrade pip

# CUDA 対応 PyTorch を明示的にインストール
RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install Cython numpy
RUN pip install einops
RUN pip install -r requirements.txt

# エントリーポイントの設定（必要に応じて変更）
CMD ["bash"]
