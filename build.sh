#!/bin/bash
set -e

IMAGE_NAME="nyc326/dsba_pretrain"
TAG="nlp"
echo "Building image $IMAGE_NAME:$TAG"
docker build -t $IMAGE_NAME:$TAG .
echo "Image $IMAGE_NAME:$TAG built successfully"