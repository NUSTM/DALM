# -*- coding: utf-8 -*-

import argparse
import io
import math
import os
import logging
import random
import time
import pickle
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import AdamW, BertPreTrainedModel, BertModel, BertTokenizer

import models
from models import BERT_CRF
import data_utils
from data_utils import OurDataset
from optimization import BertAdam
import eval_utils


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def initial_setting():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='absa', type=str,
                        help="The name of the task")
    parser.add_argument("--domain_pair", default='rest-laptop', type=str,
                        help="The name of the dataset, selected from: [rest, device, laptop, service]")
    parser.add_argument("--model_name_or_path", default='bert-cross', type=str,
                        help="Path to pre-trained model or shortcut name")              
    parser.add_argument("--output_dir", default='./pseudo_outputs/', type=str,
                        help="Path to results and models")
    parser.add_argument("--do_filter", action='store_true', help="Whether to filter samples using based model.")


    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    args = parser.parse_args()

    print(args.domain_pair)

    args.source, args.target = args.domain_pair.split('-')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    dataset_dir = f"{args.output_dir}/{args.domain_pair}"
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    args.output_dir = dataset_dir

    return args


def do_filter(args, logger):
    print('################## Doing Filter ##################')

    label_list = data_utils.get_labels(args.task)

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    data_path = f'{args.output_dir}/gen.txt'
    dataset = OurDataset(data_path=data_path, task=args.task, tokenizer=tokenizer, max_len=args.max_seq_length)
    test_loader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=4)
    
    model = torch.load(f"./pseudo_outputs/{args.source}-{args.target}/model.pt")
    model.cuda()
    model.eval()

    all_predict_label_ids, all_gold_label_ids = [], []
    for batch in tqdm(test_loader):

        input_ids, input_masks, label_ids = batch["input_id"].cuda(), batch["input_mask"].cuda(), batch["label_id"].cuda()
        predict_logits = model(input_ids, input_masks)

        predict_label_ids = [logit[1:-1] for logit in predict_logits]

        gold_label_ids, all_mask = label_ids.tolist(), input_masks.tolist()
        new_gold_label_ids = []
        for i in range(len(all_mask)):
            l = sum(all_mask[i])
            new_gold_label_ids.append(gold_label_ids[i][1:l-1])

            assert len(predict_label_ids[i]) == len(new_gold_label_ids[i]), print(len(predict_label_ids[i]), len(new_gold_label_ids[i]))
        
        all_predict_label_ids.extend(predict_label_ids)
        all_gold_label_ids.extend(new_gold_label_ids)
    

    token_list, pre_list, gold_list = [], [], []
    raw_sentences, _ = data_utils.read_generated_data(data_path)
    for i in range(len(all_predict_label_ids)):
        sentence = raw_sentences[i]
        token_a = []
        for t in [word.lower() for word in sentence]:
            token_a.extend(tokenizer.tokenize(t))
        token_list.append(token_a[:args.max_seq_length-2])

        pre_list.append([label_list[a] for a in all_predict_label_ids[i]])
        gold_list.append([label_list[a] for a in all_gold_label_ids[i]])

        assert len(pre_list[i]) == len(token_list[i]), print(len(pre_list[i]), len(token_list[i]))

    filtered_data_num = 0
    with open(os.path.join(args.output_dir, 'filter.txt'), 'w') as fw:
        for i in range(len(pre_list)):
            if ' '.join(pre_list[i]) == ' '.join(gold_list[i]):
                ts, ls = data_utils.convert_raw_data(token_list[i], pre_list[i])
                line = ' '.join(ts) + '####' + ' '.join(ls)
                fw.write(line + '\n')
                filtered_data_num += 1
    logger.info('Data length after done filter = %d', filtered_data_num)


if __name__ == '__main__':
    
    args = initial_setting()

    logger = get_logger(os.path.join(args.output_dir, f'logging_file.log'))

    if args.do_filter:
        do_filter(args, logger)