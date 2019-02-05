#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
$ nvidia-docker run -it --rm --name tmp -v `pwd`/models/topic_extractor:/workspace/topic_extractor -v `pwd`/simNet:/workspace/simNet -v `pwd`/data:/workspace/data nomotoeriko/visual_concepts:latest /bin/bash
(container)$ python get_topics.py --coco_images data/coco --output image_topics.json
"""

import _init_paths
import caffe, test_model, cap_eval_utils, sg_utils as utils
import cv2, numpy as np
import argparse
from os.path import isfile, isdir, join
from os import listdir
from test_model import upsample_image
import re
import sys
import json
from pprint import pprint


MESSAGES = {
    "isfile": "ERROR: No such file. %s",
    "isdir": "ERROR: No such directory. %s",
    "not_isfile": "WORNING: %s is already exists. Continue? [[y], n]",
    "topic_num": "WORNING: Topic num != %d.\n\tImage_id: %d\n\tTopics: %s"
}


class TopicExtractor:

    def __init__(self, args):
        assert isdir(args.coco), MESSAGES["isdir"] % args.coco
        assert isdir(join(args.coco, "train2014")), MESSAGES["isdir"] % join(args.coco, "train2014")
        assert isdir(join(args.coco, "val2014")), MESSAGES["isdir"] % join(args.coco, "val2014")
        assert isdir(join(args.coco, "test2014")), MESSAGES["isdir"] % join(args.coco, "test2014")
        self.coco = args.coco
        if isfile(args.output):
            comm = input(MESSAGES["not_isfile"] % args.output)
            if "n" in comm:
                print("Kill")
                exit(0)

        vocab_file = args.vocab
        assert isfile(vocab_file), MESSAGES["isfile"] % vocab_file
        self.vocab = utils.load_variables(vocab_file)

        # Set up Caffe
        caffe.set_mode_gpu()
        caffe.set_device(0)

        # Load the model
        mean = np.array([[[103.939, 116.779, 123.68]]])
        base_image_size = 565
        prototxt_deploy = args.prototxt
        assert isfile(prototxt_deploy), MESSAGES["isfile"] % prototxt_deploy
        model_file = args.model
        assert isfile(model_file), MESSAGES["isfile"] % model_file
        assert isdir(model_file + "_output"), MESSAGES["isdir"] % (model_file + "_output")
        self.model = test_model.load_model(prototxt_deploy, model_file, base_image_size, mean, self.vocab)

        # define functional words
        self.functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is',
                                 'to', 'an', 'two', 'at', 'next', 'are', "that", "it"]
        self.is_functional = np.array([x not in self.functional_words for x in self.vocab['words']])

        # load the score precision mapping file
        eval_file = model_file + '_output/coco_valid1_eval.pkl'
        assert isfile(eval_file), MESSAGES["isfile"] % eval_file
        self.pt = utils.load_variables(eval_file)

        self.filename_template = re.compile(r"COCO_(train|val|test)2014_(\d{12}).jpg")
        self.bs = args.bs
        self.topic_num = args.topic_num
        self.tgt = args.tgt

    def topic_extract(self):
        """
        Extraxt the topics from image list
        :return: topic list [{image_id: 9, image_concepts: [dog, cat, ...]}, ...]
        """
        image_topics = []
        image_id_batch = []
        image_batch = []
        # For val data
        if "val" in self.tgt:
            print("val2014")
            for image_file in listdir(join(self.coco, "val2014")):
                m = self.filename_template.match(image_file)
                if m:
                    image_id = int(m.group(2))
                    image_id_batch.append(image_id)

                    im = cv2.imread(join(self.coco, "val2014", image_file))
                    image_batch.append(im)

                if len(image_id_batch) == self.bs:
                    sys.stdout.write("*")
                    sys.stdout.flush()
                    prec = self.__mk_prec_from_batch(image_batch)
                    for i in range(self.bs):
                        topics = self.__output_words_image(prec[i, :])
                        im_id = image_id_batch[i]
                        if len(topics) != self.topic_num:
                            print(MESSAGES["topic_num"] % (self.topic_num, im_id, " ".join(topics)))
                        image_topics.append(
                            {"image_id": im_id, "image_concepts": topics}
                        )
                    # Delete current batch
                    image_id_batch = []
                    image_batch = []

        # For test data
        if "test" in self.tgt:
            print("test2014")
            for image_file in listdir(join(self.coco, "test2014")):
                m = self.filename_template.match(image_file)
                if m:
                    image_id = int(m.group(2))
                    image_id_batch.append(image_id)

                    im = cv2.imread(join(self.coco, "test2014", image_file))
                    image_batch.append(im)

                if len(image_id_batch) == self.bs:
                    sys.stdout.write("*")
                    sys.stdout.flush()
                    prec = self.__mk_prec_from_batch(image_batch)
                    for i in range(self.bs):
                        topics = self.__output_words_image(prec[i, :])
                        im_id = image_id_batch[i]
                        if len(topics) != self.topic_num:
                            print(MESSAGES["topic_num"] % (self.topic_num, im_id, " ".join(topics)))
                        image_topics.append(
                            {"image_id": im_id, "image_concepts": topics}
                        )
                    # Delete current batch
                    image_id_batch = []
                    image_batch = []
 
        # For train (if val data remains, mixed)
        if "train" in self.tgt:
            print("\ntrain2014")
            for image_file in listdir(join(self.coco, "train2014")):
                m = self.filename_template.match(image_file)
                if m:
                    image_id = int(m.group(2))
                    image_id_batch.append(image_id)

                    im = cv2.imread(join(self.coco, "train2014", image_file))
                    image_batch.append(im)

                if len(image_id_batch) == self.bs:
                    sys.stdout.write("*")
                    sys.stdout.flush()
                    prec = self.__mk_prec_from_batch(image_batch)
                    for i in range(self.bs):
                        topics = self.__output_words_image(prec[i, :])
                        im_id = image_id_batch[i]
                        if len(topics) != self.topic_num:
                            print(MESSAGES["topic_num"] % (self.topic_num, im_id, " ".join(topics)))
                        image_topics.append(
                            {"image_id": im_id, "image_concepts": topics}
                        )
                    # Delete current batch
                    image_id_batch = []
                    image_batch = []

        # For remaining batch
        if image_id_batch:
            # Pad batch
            tid = None
            timage = np.zeros((self.model["base_image_size"], self.model["base_image_size"], 3))
            padnum = self.bs - len(image_id_batch)
            image_id_batch += [tid] * padnum
            image_batch += [timage] * padnum

            # Topic extraction
            prec = self.__mk_prec_from_batch(image_batch)
            for i, im_id in enumerate(image_id_batch):
                if not im_id:
                    break
                topics = self.__output_words_image(prec[i, :])
                if len(topics) != self.topic_num:
                    print(MESSAGES["topic_num"] % (self.topic_num, im_id, " ".join(topics)))
                image_topics.append(
                    {"image_id": im_id, "image_concepts": topics}
                )
        return image_topics

    def __preprocess_batch(self, image_batch):
        """
        Preprocess of images
        :param image_batch: cv2 read image list
        :return: preprocessed image batch
        """
        images = [upsample_image(np.array(im, dtype=np.float32) - self.model["means"], self.model['base_image_size'])[0]
                  for im in image_batch]
        images = np.array(images, dtype=np.float32).transpose((0, 3, 1, 2))
        return images

    def __mk_prec_from_batch(self, image_batch):
        # Topic Extract
        images = self.__preprocess_batch(image_batch)
        net = self.model["net"]
        net.forward(data=images.astype(np.float32, copy=False))
        mil_prob = net.blobs['mil'].data.copy()

        mil_prob = mil_prob.reshape(self.bs, mil_prob.size // self.bs)

        # Compute precision mapping
        prec = np.zeros(mil_prob.shape)
        for jj in range(prec.shape[1]):
            prec[:, jj] = cap_eval_utils.compute_precision_score_mapping(
                self.pt['details']['score'][:, jj] * 1,
                self.pt['details']['precision'][:, jj] * 1,
                mil_prob[:, jj] * 1
            )
        return prec

    def __output_words_image(self, p):
        ind_output = np.argsort(p)
        ind_output = ind_output[::-1]
        topics = []
        for ind in ind_output:
            if self.is_functional[ind]:
                topics.append(self.vocab["words"][ind])
                if len(topics) == self.topic_num:
                    break
        return topics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_images", default="./data/coco",
                        type=str, dest="coco")
    parser.add_argument("--output", default="image_topics.json",
                        type=str, dest="output")
    parser.add_argument("--tgt", default="val_train",
                        type=str, dest="tgt")
    # -------------------------Model Settings---------------------------------------------------------
    parser.add_argument("--vocab_file", default="./code/vocabs/vocab_train.pkl",
                        type=str, dest="vocab")
    parser.add_argument("--prototxt_deploy", default="./code/output/vgg/mil_finetune.prototxt.deploy",
                        type=str, dest="prototxt")
    parser.add_argument("--model", default="./topic_extractor/snapshot_iter_240000.caffemodel",
                        type=str, dest="model")
    parser.add_argument("--batch_size", default=32,
                        type=int, dest="bs")
    parser.add_argument("--topic_num", default=10,
                        type=int, dest="topic_num")
    args = parser.parse_args()
    print("Settings:")
    pprint(args)

    # Topic extraction
    e = TopicExtractor(args)
    topic_list = e.topic_extract()

    # Create output file
    with open(args.output, "w") as f:
        json.dump(topic_list, f)
    print("Create topic bag %s" % args.output)
