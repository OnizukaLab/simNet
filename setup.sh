#!/usr/bin/env bash
# nomotoeriko/simnet に合わせて作成

python setup.py
cp /root/.torch/models/resnet152-b121ed2d.pth /workspace/models/  # copy to volume mounted dir
