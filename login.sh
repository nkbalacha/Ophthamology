#!/bin/bash

cid=`docker ps | grep nitinkb/pytorch:vivaldi | awk '{print $1}'`
# cid=`docker ps | grep neerajwagh/opencl:test | awk '{print $1}'`

docker exec -ti $cid /bin/bash


