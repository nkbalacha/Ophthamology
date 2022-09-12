#!/bin/bash

docker build -t nitinkb/pytorch:vivaldi -f Dockerfile.cu11.1 .
docker push nitinkb/pytorch:vivaldi

# docker build -t neerajwagh/opencl:test -f Dockerfile.opencl .
# docker push neerajwagh/opencl:test