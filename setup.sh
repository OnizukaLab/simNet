#!/usr/bin/env bash
# nomotoeriko/simnet に合わせて作成
# 最初に一回だけやればいい
# -v `pwd`/data/coco:/workspace/data -v `pwd`/simNet:/workspace/simNet -v `pwd`/models:/workspace/models

python simNet/KarpathySplit.py
python simNet/build_vocab.py

python simNet/setup.py
cp /root/.torch/models/resnet152-b121ed2d.pth /workspace/models/  # copy to volume mounted dir

# cp models/vocab.pkl ./
# train
# python simNet/train.py --model_path /workspace/models/ --pretrained_cnn /workspace/models/resnet152-b121ed2d.pth --image_dir ./data
