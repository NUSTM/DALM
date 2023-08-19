import os
import sys

sys.path.append('../')
import logging
import argparse
import random
import json
import data_utils
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import GPT2Tokenizer,GPT2LMHeadModel,AdamW
import math
from collections import namedtuple
from tqdm import tqdm

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


class GPT_label(nn.Module):
    def __init__(self,label_size=8):
        super(GPT_label, self).__init__()
        self.labels=['O', 'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU','PAD']
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpt=GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
        self.embed_label=nn.Embedding(label_size,768)
        self.fc_label=nn.Linear(768,label_size)
        

    def forward(self,text,label):
        label_hiden=self.embed_label(label) #32,99,768
        text_hiden=self.gpt.transformer(text)[0] # 32,99,768
        fusion_hiden=label_hiden+text_hiden

        label_logits=self.fc_label(fusion_hiden) # 32,99,8
        text_logits=self.gpt.lm_head(fusion_hiden) #32,99,50257

        return label_logits,text_logits
    

    def generate(self, max_seq):
        label_dict = {}
        for (i, labe) in enumerate(self.labels):
            label_dict[labe] = i

        gpt_text = ['<|endoftext|>', 'Á'] 
        gpt_label = ['O', 'O'] 

        text_id = self.tokenizer.convert_tokens_to_ids(gpt_text)
        label_id = [label_dict[i] for i in gpt_label]

        out_label=[]
        out_text = []

        for t in range(max_seq):
            inp_tag = torch.tensor(label_id).to(self.device).unsqueeze(0)
            inp_text = torch.tensor(text_id).to(self.device).unsqueeze(0)

            label_logits, text_logits=self.forward(inp_text,inp_tag)

            tag = torch.argmax(label_logits, -1)[0,-1].item()
       
            text_logits=text_logits[:, -1, :]
            word_weights=text_logits.softmax(-1) # 1,50257

            top_n = word_weights.topk(100)[1] # 1,100
            word_mask = word_weights.scatter(-1, top_n, 1000000001)
            weight_mask = word_mask < 1000000000
            word_weights = word_weights.masked_fill(weight_mask, 0)
            word_indices = torch.multinomial(word_weights, 1)

            word_indices = word_indices.squeeze(1).item()
            # print(word_indices.size())
            label_id.append(tag)
            text_id.append(word_indices)
            if word_indices==50256:
                # <|endoftext|>
                out_label.append(tag)
                break

            if t!=0:
                out_label.append(tag)
            out_text.append(word_indices)

        return out_text, out_label


def train(args,logger):
    processor = data_utils.ABSAProcessor()
    label_list = processor.get_labels()
    model = GPT_label()
    if torch.cuda.is_available():
        model=model.cuda()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    train_examples = processor.get_train_examples(args.input_dir)

    train_features = data_utils.convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)

    num_train_steps = int(math.ceil(len(train_features) / args.train_batch_size)) * args.num_train_epochs

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    text_loss_ft=nn.CrossEntropyLoss(ignore_index=129)
    label_loss_ft = nn.CrossEntropyLoss(ignore_index=7)

    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,)

    train_steps = len(train_dataloader)
    max_loss_val=99999
    model.train()
    for e_ in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        loss_data=0
        for step in range(train_steps):
            batch = train_iter.next()
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
            else:
                batch = tuple(t for t in batch)
            input_ids, label_ids = batch

            label_logits, text_logits = model(input_ids[:,:-1], label_ids[:,:-2])
            label_loss = label_loss_ft(label_logits.contiguous().view(-1, 8), label_ids[:, 1:-1].contiguous().view(-1))
            text_loss = text_loss_ft(text_logits.contiguous().view(-1, 50257), input_ids[:, 1:].contiguous().view(-1))
            loss=text_loss+label_loss
            if step % 50==0:
                logger.info('[epoch: %d], [step: %d], the loss is %f, the text_loss is %f, the label_loss is %f',
                            e_,step,loss.item(),text_loss.item(),label_loss.item())
            loss_data+=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if loss_data<max_loss_val:
            logger.info('[epoch: %d] , the loss is %f', e_,loss_data)
            max_loss_val=loss_data
            torch.save(model, os.path.join(args.model_dir, f'{args.source_domain}-{args.target_domain}.pt'))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        default='./process_data/service-rest/final_train.txt',
                        type=str,
                        required=False,
                        help="The input data dir containing json files.")
    
    parser.add_argument("--model_dir",
                        default='./GPT_based/models/',
                        type=str,
                        required=False,
                        help="The input data dir containing json files.")
    parser.add_argument("--source_domain", default='service',type=str,required=False)
    parser.add_argument("--target_domain", default='rest',type=str,required=False)

    parser.add_argument('--seed',
                        type=int,
                        default=62,
                        help="random seed for initialization")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=100,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--learning_rate",
                        default=3e-4,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.model_dir, f'{args.source_domain}-{args.target_domain}.log'))

    train(args,logger)


if __name__ == "__main__":
    main()