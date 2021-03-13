#!/bin/bash

set -e
set -x

docker build -t ykilcher/jannet .
docker push ykilcher/jannet
