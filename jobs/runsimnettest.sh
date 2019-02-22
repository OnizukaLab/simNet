#!/bin/bash

PYTHONPATH=/home/aca10405ri:/workspace/ python simNet/test.py --trained models/simNet-30.pkl\
  --vocab_path vocab.pkl --image_dir data/coco\
  --caption_test_path data/annotations/karpathy_split_test.json --topic_path data/coco/visual_concepts/image_topics.json

