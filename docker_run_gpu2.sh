#!/bin/bash

# イメージ名
IMAGE_NAME="opengait"

# コンテナ名
CONTAINER_NAME="opengait_container_2"

# ホスト側の作業ディレクトリ（必要に応じて調整）
HOST_WORKDIR=$(pwd)

# コンテナ内の作業ディレクトリ
CONTAINER_WORKDIR="/app/OpenGait"

# Docker コンテナを実行
#GPU
docker run --gpus '"device=2"' --rm -it \
  --name $CONTAINER_NAME \
  -v "$HOST_WORKDIR":"$CONTAINER_WORKDIR" \
  -v /home/ryu/OpenGait/CASIA-B:/app/OpenGait/CASIA-B \
  -v /home/ryu/OpenGait/CASIA-B-png:/app/OpenGait/CASIA-B-png \
  -w "$CONTAINER_WORKDIR" \
  --env NCCL_DEBUG=INFO \
  --env NCCL_IB_DISABLE=1 \
  $IMAGE_NAME \
  "$@"
