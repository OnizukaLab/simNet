#!/usr/bin/env bash
# nomotoeriko/simnet に合わせて作成
# -v `pwd`/data/coco:/workspace/data -v `pwd`/simNet:/workspace/simNet -v `pwd`/models:/workspace/model

python simNet/KarpathySplit.py
python3 simNet/build_vocab.py

python simNet/setup.py
cp /root/.torch/models/resnet152-b121ed2d.pth /workspace/models/  # copy to volume mounted dir
