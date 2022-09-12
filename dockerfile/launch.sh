#!/bin/bash

docker run --rm --gpus=all --shm-size 90G --ipc=host -v /home/nitin:/home/nitin -v /mnt/scratch:/mnt/scratch -v /shared:/shared -v /scr:/scr -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -w /home/nitin --user="$(id -u):$(id -g)" --volume="$PWD:/app" -p 8988:8988 -p 7178:7178 -p 6106:6106 -p 6107:6107 -p 6108:6108 -p 6109:6109 -p 8150:8150 -it nitinkb/pytorch:vivaldi

# docker run --rm --gpus=all --shm-size 90G --ipc=host -v /home/neeraj:/home/neeraj -v /mnt/scratch:/mnt/scratch -v /shared:/shared -v /scr:/scr -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -w /home/neeraj --user="$(id -u):$(id -g)" --volume="$PWD:/app" -p 8888:8888 -p 7078:7078 -p 6006:6006 -p 6007:6007 -p 6008:6008 -p 6009:6009 -p 8050:8050 -it neerajwagh/opencl:test