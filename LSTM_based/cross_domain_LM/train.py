import os
from pprint import pformat
import sys
import time

sys.path.append('../')
import logging
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm

import data_utils
from data_utils import LMDataset
from model import LMModel


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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='./process_data/service-rest/final_train.txt', type=str)
    parser.add_argument("--domain_pair", default='service-rest', type=str)
    parser.add_argument("--output_dir", default='./GPT_models/', type=str)

    # model para
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--bidirectional_encoder", action="store_true")
    parser.add_argument("--rnn_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=-1)
    parser.add_argument("--num_enc_layers", type=int, default=1)
    parser.add_argument("--num_dec_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--param_init", type=float, default=0.1)
    parser.add_argument("--num_z_samples", type=int, default=0)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--z_cat", action="store_true")
    parser.add_argument("--use_avg", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--word_dropout_rate", type=float, default=0.0)
    parser.add_argument("--inputless", action="store_true")
    # optimizer
    parser.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    # others
    parser.add_argument("--sent_length_trunc", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--tol", type=int, default=0.01)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--report_every", type=int, default=100)
    parser.add_argument("--gpuid", type=int, default=0)

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


def train(args, logger):
    data_utils.set_params(args)
    logger.info("Config:\n%s", pformat(vars(args)))


    fields = data_utils.build_fields()
    logger.info("Fields: %s", fields.keys())

    logger.info("Load %s", args.input_file)
    train_data = LMDataset(fields, args.input_file)
    logger.info("Training sentences: %d", len(train_data))

    fields["sent"].build_vocab(train_data)
    fields["label"].build_vocab(train_data)
    vocab, label_vocab = fields["sent"].vocab, fields["label"].vocab

    train_iter = data_utils.build_dataset_iter(train_data, args)

    model = LMModel(fields, args)
    model.to(args.device)
    logger.info("Model:\n%s", model)

    text_loss_ft=nn.CrossEntropyLoss(ignore_index=vocab.stoi[data_utils.PAD_WORD])
    label_loss_ft = nn.CrossEntropyLoss(ignore_index=label_vocab.stoi[data_utils.PAD_LABEL])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    """Try to train and validate given a number of epochs."""
    start_time = time.time()
    logger.info("Start training")
    try:
        model.train()
        max_loss_val=99999
        for epoch in range(args.start_epoch, args.epochs + 1):
            loss_data=0
            for step, batch in enumerate(train_iter):

                text = batch.sent
                label = batch.label

                text_logits, label_logits = model(text[:-1],label[:-2])
                text_loss = text_loss_ft(text_logits.contiguous().view(-1, len(vocab)), text[1:, :].contiguous().view(-1))
                label_loss = label_loss_ft(label_logits.contiguous().view(-1, len(label_vocab)), label[1:-1, :].contiguous().view(-1))
                loss = text_loss + label_loss

                loss_data += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if step%50==0:
                    logger.info('[epoch: %d], [step: %d], the loss is %f, the text_loss is %f, the label_loss is %f',
                                epoch,step,loss.item(),text_loss.item(),label_loss.item())
                    
            if loss_data < max_loss_val:
                logger.info('[epoch: %d] , the loss is %f', epoch,loss_data)
                max_loss_val = loss_data
                data_utils.save_model(fields, model, optimizer, epoch, args)

    except KeyboardInterrupt:
        logger.info("Training interupted")

    logger.info("End of training: time %.1f min", (time.time() - start_time) / 60)


if __name__ == "__main__":
    args = initial_setting()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger = get_logger(f'{args.output_dir}/logging.log')

    train(args,logger)
