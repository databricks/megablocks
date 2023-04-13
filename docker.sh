#!/bin/bash

BASEDIR=`cd .. && pwd`
sudo docker run \
     --cap-add SYS_ADMIN \
     --ipc=host \
     --runtime=nvidia \
     --network=host \
     --privileged \
     -it \
     -v${BASEDIR}:/mount \
     -v/future/u/tgale/data/pile_gpt2:/tmp \
     megablocks-dev:latest
