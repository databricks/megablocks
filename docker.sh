#!/bin/bash

BASEDIR=`cd .. && pwd`
docker run \
       --cap-add SYS_ADMIN \
       --ipc=host \
       --runtime=nvidia \
       -it \
       -v${BASEDIR}:/mount \
       megablocks-dev:latest
