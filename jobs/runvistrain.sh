#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/workspace/code
python get_topics.py --coco_images data/coco --output data/coco/visual_concepts/image_topics_train.json\
 --vocab_file /workspace/code/vocabs/vocab_train.pkl\
 --prototxt_deploy ./visual_concepts_model/vgg/mil_finetune.prototxt.deploy\
 --model visual_concepts_model/vgg/snapshot_iter_240000.caffemodel --batch_size 24 --tgt train
