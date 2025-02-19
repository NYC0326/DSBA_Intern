#!/bin/bash

CONTAINER_NAME="nlp"
echo "Starting container $CONTAINER_NAME"
docker exec -it $CONTAINER_NAME /bin/bash