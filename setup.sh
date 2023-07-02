#!/bin/bash
set -euxo pipefail

python3 -m venv venv
. venv/bin/activate

apt-get install -yq --no-install-recommends python3-pip \
        python3-dev \
        wget \
        git  \
        libopencv-dev \
        tzdata && apt-get upgrade -y && apt-get clean

pip install -U pip && pip install --no-cache-dir torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install --upgrade --no-cache-dir -r ./requirements.txt