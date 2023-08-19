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
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")        
    parser.add_argument("--do_pseudo_labeling", action='store_true', help="Whether to assign pseudo label.")        
    parser.add_argument("--do_eval", action='store_true', help="Whether to run direct eval on the test set.")


    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5 , type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=62, help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args, logger):
    print('training')
    label_list = data_utils.get_labels(args.task)
    model = BERT_CRF.from_pretrained(args.model_name_or_path, num_labels = len(label_list))

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    # get source&target data
    source_data_path = f'./raw_data/{args.source}_train.txt'
    source_dataset = OurDataset(data_path=source_data_path, task=args.task, tokenizer=tokenizer, max_len=args.max_seq_length)
    target_data_path = f'./tuple_dp_op/{args.target}.txt'
    target_dataset = OurDataset(data_path=target_data_path, task=args.task, tokenizer=tokenizer, max_len=args.max_seq_length)

    # unify num of source data and target data
    source_dataset, target_dataset = data_utils.unify_data_num(source_dataset, target_dataset)
    logger.info('Data length = %d', len(source_dataset))

    # data_loader
    source_dataloader = DataLoader(source_dataset, batch_size=args.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
    target_dataloader = DataLoader(target_dataset, batch_size=args.train_batch_size, drop_last=True, shuffle=True, num_workers=4)

    model.cuda()
    
    # optimeizer
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_steps = int(math.ceil(len(source_dataset) / args.train_batch_size)) * args.num_train_epochs
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps
                         )

    train_steps = len(source_dataloader)
    model.train()

    for epoch in range(args.num_train_epochs):

        source_iter, target_iter = iter(source_dataloader), iter(target_dataloader)
        for step in range(train_steps):
            source_batch, target_batch = source_iter.next(), target_iter.next()
            source_input_ids, source_input_masks, source_label_ids = source_batch["input_id"].cuda(), source_batch["input_mask"].cuda(), source_batch["label_id"].cuda()
            target_input_ids, target_input_masks, target_label_ids = target_batch["input_id"].cuda(), target_batch["input_mask"].cuda(), target_batch["label_id"].cuda()

            # calculate absa loss on source domain data and token-level representation
            source_output, absa_loss = model(source_input_ids, attention_mask=source_input_masks, labels=source_label_ids)
            target_output, _ = model(target_input_ids, attention_mask=target_input_masks, labels=target_label_ids)

            # calcualte aspect-level MMD loss between source and target domain data
            source_aspect_rep = models.get_aspect_rep(source_output,source_label_ids)
            target_aspect_rep = models.get_aspect_rep(target_output, target_label_ids)
            if int(source_aspect_rep.shape[0])==0 or int(target_aspect_rep.shape[0])==0:
                continue
            mmd_loss = models.cal_MMD(source_aspect_rep, target_aspect_rep)

            loss = absa_loss + 0.01*mmd_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 25 == 0:
                logger.info('Epoch: %d, batch: %d, absa loss: %f, mmd loss: %f, loss: %f', 
                            epoch, step, absa_loss, mmd_loss, loss)

    torch.save(model, os.path.join(args.output_dir, f"model.pt"))


def pseudo_labeling(args, logger):
    print("assgning pseduo label")
    label_list = data_utils.get_labels(args.task)

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    # load data
    data_path = f'./raw_data/{args.target}_train.txt'
    dataset = OurDataset(data_path=data_path, task=args.task, tokenizer=tokenizer, max_len=args.max_seq_length)
    test_loader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=4)

    logger.info('Data length = %d', len(dataset))
    
    # load model
    model = torch.load(os.path.join(args.output_dir, f"model.pt"))
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

    # get and save token_list and predicted sequence
    token_list, pre_list, gold_list = [], [], []
    raw_sentences, _ = data_utils.read_absa_raw_data(data_path)
    for i in range(len(all_predict_label_ids)):
        sentence = raw_sentences[i]
        token_a = []
        for t in [word.lower() for word in sentence]:
            token_a.extend(tokenizer.tokenize(t))
        token_list.append(token_a[:args.max_seq_length-2])

        pre_list.append([label_list[a] for a in all_predict_label_ids[i]])
        gold_list.append([label_list[a] for a in all_gold_label_ids[i]])

        assert len(pre_list[i]) == len(token_list[i]), print(len(pre_list[i]), len(token_list[i]))

    with open(os.path.join(args.output_dir, f'pre.txt'), 'w') as fw:
        lines = [' '.join(token_list[i]) + '####' + ' '.join(pre_list[i]) for i in range(len(pre_list))]
        fw.write('\n'.join(lines))


def eval(args, logger):
    print("evaluation")
    label_list = data_utils.get_labels(args.task)

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    # load data
    data_path = f'./raw_data/{args.target}_test.txt'
    dataset = OurDataset(data_path=data_path, task=args.task, tokenizer=tokenizer, max_len=args.max_seq_length)
    test_loader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=4)

    logger.info('Data length = %d', len(dataset))
    
    # load model
    model = torch.load(os.path.join(args.output_dir, f"model.pt"))
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

    # get and save token_list and predicted sequence
    token_list, pre_list, gold_list = [], [], []
    raw_sentences, _ = data_utils.read_absa_raw_data(data_path)
    for i in range(len(all_predict_label_ids)):
        sentence = raw_sentences[i]
        token_a = []
        for t in [word.lower() for word in sentence]:
            token_a.extend(tokenizer.tokenize(t))
        token_list.append(token_a[:args.max_seq_length-2])

        pre_list.append([label_list[a] for a in all_predict_label_ids[i]])
        gold_list.append([label_list[a] for a in all_gold_label_ids[i]])

        assert len(pre_list[i]) == len(token_list[i]), print(len(pre_list[i]), len(token_list[i]))

    with open(os.path.join(args.output_dir, f'pre.txt'), 'w') as fw:
        lines = [' '.join(token_list[i]) + '####' + ' '.join(pre_list[i]) for i in range(len(pre_list))]
        fw.write('\n'.join(lines))

    # calculate ate results (pre, Recall, F1)
    ate_precision, ate_recall, ate_f1 = eval_utils.eval_ate_results(pre_list, gold_list)
    logger.info('ATE Task: Precision = %f, Recall = %f, F1 = %f', ate_precision, ate_recall, ate_f1)
    # calculate absa results (pre, Recall, F1)
    absa_precision, absa_recall, absa_f1 = eval_utils.eval_absa_results(pre_list, gold_list)
    logger.info('ABSA Task: Precision = %f, Recall = %f, F1 = %f', absa_precision, absa_recall, absa_f1)


if __name__ == '__main__':
    
    args = initial_setting()

    set_seed(args.seed)

    logger = get_logger(os.path.join(args.output_dir, f'logging_file.log'))

    if args.do_train:
        train(args, logger)

    if args.do_pseudo_labeling:
        pseudo_labeling(args, logger)

    if args.do_eval:
        eval(args, logger)