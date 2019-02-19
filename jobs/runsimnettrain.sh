#!/bin/bash

python simNet/train.py --model_path ~/models --vocab_path ~/vocab.pkl\
 --image_dir ~/data/coco --caption_path ~/data/annotations/karpathy_split_train.json\
 --topic_path ~/data/coco/visual_concepts/image_topics.json --pretrained_cnn ~/.torch/models/resnet152-b121ed2d.pth
