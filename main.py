# -*- coding: utf-8 -*-
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pickle
from build_vocab import Vocabulary
from model import Encoder2Decoder
from torch.autograd import Variable
from torchvision import transforms, datasets
from coco.pycocotools.coco import COCO
from coco.pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt

from .test import to_var, CocoEvalLoader

def main(args):
    # Load vocabulary wrapper.
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    # Load trained model
    model = Encoder2Decoder(args.embed_size, len(vocab), args.hidden_size)
    model.load_state_dict(torch.load(args.trained))

    # Change to GPU mode is available
    if torch.cuda.is_available():
        print("DEVICE COUNT: ", torch.cuda.device_count())
        model.cuda()

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Wrapper the COCO VAL dataset
    eval_data_loader = torch.utils.data.DataLoader(
        CocoEvalLoader(args.image_dir, args.caption_test_path, args.topic_path, transform),
        batch_size=args.eval_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
    epoch = int(args.trained.split('/')[-1].split('-')[1].split('.')[0])

    # Generated captions to be compared with GT
    results = []
    print('---------------------Start evaluation on MS-COCO dataset-----------------------')
    for i, (images, image_ids, _, T_val) in enumerate(eval_data_loader):

        images = to_var(images)
        T_val = to_var(T_val)
        generated_captions = model.sampler(epoch, images, T_val)

if __name__ == '__main__':
    