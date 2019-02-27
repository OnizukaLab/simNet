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

from test import to_var, CocoEvalLoader


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
        generated_captions, image_attns, topic_attns = model.sampler(epoch, images, T_val)

        if torch.cuda.is_available():
            captions = generated_captions.cpu().data.numpy()
            image_attns = image_attns.cpu().data.numpy()
            topic_attns = topic_attns.cpu().data.numpy()
        else:
            captions = generated_captions.data.numpy()
            image_attns = image_attns.data.numpy()
            topic_attns = topic_attns.data.numpy()

        # print("IMAGE ATTN. SIZE: ", image_attns.shape)
        # print("TOPIC ATTN. SIZE: ", topic_attns.shape)

        # Build caption based on Vocabulary and the '<end>' token
        for image_idx in range(captions.shape[0]):

            sampled_ids = captions[image_idx]
            sampled_image_attns = image_attns[image_idx]
            sampled_topic_attns = topic_attns[image_idx]
            sampled_caption = []
            # final_image_attns = []
            # final_topic_attns = []

            for word_id in sampled_ids:

                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                else:
                    sampled_caption.append(word)
            final_image_attns = sampled_image_attns[:len(sampled_caption)]
            final_topic_attns = sampled_topic_attns[:len(sampled_caption)]
            sentence = ' '.join(sampled_caption)

            temp = {'image_id': int(image_ids[image_idx]),
                    'caption': sentence,
                    "image_attn": final_image_attns.tolist(),  # caption len * 49
                    "topic_attn": final_topic_attns.tolist()}  # caption len * topic num
            results.append(temp)

        # Disp evaluation process
        if (i+1) % 10 == 0:
            print('[%d/%d]' % ((i+1), len(eval_data_loader)))

    print('------------------------Caption Generated-------------------------------------')
    with open(args.save_path, "wb") as f:
        pickle.dump(results, f)

    print('------------------------Results Saved-------------------------------------')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='self', help='To make it runnable in jupyter')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/coco2014',
                        help='directory for resized training images')
    parser.add_argument('--caption_test_path', type=str,
                        default='./data/annotations/karpathy_split_test.json',
                        help='path for test annotation json file')
    parser.add_argument('--topic_path', type=str,
                        default='./data/topics/image_topic.json',
                        help='path for test topic json file')

    # ---------------------------Hyper Parameter Setup------------------------------------
    parser.add_argument('--save_path', type=str, default='generated_by_main.pickle')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors, also dimension of v_g')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--trained', type=str, default='./models/simNet-30.pkl',
                        help='start from checkpoint or scratch')
    parser.add_argument('--eval_size', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    print('------------------------Model and Testing Details--------------------------')
    print(args)

    # Start main
    main(args)
