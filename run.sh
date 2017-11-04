#!/bin/bash
docker run --runtime=nvidia -v $PWD:/home/docker/mount --rm -it kbobrowski/yolo-vehicle-detection $1 $2 $3
