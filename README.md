# YOLO vehicle detection

Vehicle detection based [YOLO](https://pjreddie.com/darknet/yolo/) trained on [Udacity dataset 2](https://github.com/udacity/self-driving-car/tree/master/annotations). [Darkflow](https://github.com/thtrieu/darkflow) is used for predictions.

Algorithm attempts to stabilize detected frame around car, maintain it in case of a short break in the detection and predict position of a car hidden behind another car.

## Video

[![video](https://img.youtube.com/vi/64bETGQ-tLk/0.jpg)](https://www.youtube.com/watch?v=64bETGQ-tLk)

## Installation

There is a Docker image available, so only installation of [Nvidia drivers](https://github.com/kbobrowski/build-deep-learning-box/blob/master/build-1-nvidia-driver.sh), [Docker](https://www.docker.com/community-edition) and [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker/tree/2.0) is required. Create new directory with a video to process (e.g. [video from Udacity Self-Driving Car Nanodegree](https://github.com/kbobrowski/YOLO-vehicle-detection/raw/master/project_video.mp4)) and [YOLO weights](https://drive.google.com/open?id=0B1E_D7UxqPl2QzNYM3V4TWV6cFU) trained for one label (car).

Running following command in this new directory will generate output.mp4 file with tracked cars:

```
docker run --runtime=nvidia -v $PWD:/home/docker/mount --rm -it kbobrowski/yolo-vehicle-detection yolo-vehicles_1000.weights project_video.mp4 output.mp4
```

To access shell of the image (user password: d):
```
docker run --runtime=nvidia -v $PWD:/home/docker/mount --rm -it --entrypoint /bin/bash kbobrowski/yolo-vehicle-detection
```
