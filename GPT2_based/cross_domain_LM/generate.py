"""Generating module"""
# from __future__ import print_function
import argparse
import io
import math
from pprint import pformat
from random import random
import torch
from transformers import GPT2Tokenizer
from train import GPT_label
import random
from tqdm import tqdm
import os

dic = {0:'O', 1:'B-POS', 2:'B-NEG', 3:'B-NEU', 4:'I-POS', 5:'I-NEG', 6:'I-NEU',7:'PAD'}

def save(samples, samples_label, out_file):
    """Save samples to file."""
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2')

    with io.open(out_file, "w", encoding="utf-8", errors="ignore") as f:
        for index, sample in enumerate(samples):
            gpt_tokens=tokenizer.convert_ids_to_tokens(sample)
            gpt_labels=[dic[i] for i in samples_label[index]]
            tokens, labels = [], []
            
            if len(gpt_labels)!=len(gpt_tokens):
                continue

            for i, token in enumerate(gpt_tokens):
                if i==0:
                    tokens.append(token)
                    labels.append(gpt_labels[i])
                elif token[0]=='Ġ':
                    tokens.append(token[1:])
                    labels.append(gpt_labels[i])
                else: tokens[-1]+=token

            if (len(tokens) == len(labels)):
                f.write(u" ".join(tokens) + '####' + " ".join(labels))
                f.write(u"\n")
                f.flush()


def main():
    """Main workflow"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--source",
                        default='service',
                        type=str,
                        required=False,
                        help="The input data dir containing json files.")
    parser.add_argument("--target",
                        default='rest',
                        type=str,
                        required=False,
                        help="The input data dir containing json files.")


    ## Other parameters
    parser.add_argument("--max_seq",
                        default=100,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--generate_number",
                        default=10000,
                        type=int)


    args = parser.parse_args()

    print(args.source + "====================================>" + args.target)

    model_path = './GPT2_based/models/'+ args.source + '-' + args.target +'.pt'

    output_path = './GPT2_based/generated_data/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = output_path + args.source + '-' + args.target
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = output_path + '/gen.txt'


    model=torch.load(model_path)
    samples = []
    samples_label = []
    with torch.no_grad():
        for i in tqdm(range(args.generate_number)):
            new_batch, new_label = model.generate(args.max_seq)
    
            samples_label.append(new_label)
            samples.append(new_batch)
    
    save(samples, samples_label,output_path)


if __name__ == "__main__":
    main()
