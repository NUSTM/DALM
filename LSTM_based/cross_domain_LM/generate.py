"""Generating module"""
# from __future__ import print_function
import argparse
import io
import math
from pprint import pformat
from random import random
import torch
import random
from tqdm import tqdm
import os

import data_utils
from model import LMModel


def save(samples, samples_label, fields, output_path):
    """Save samples to file."""
    vocab = fields["sent"].vocab
    label_vocab = fields["label"].vocab
    with io.open(output_path, "w", encoding="utf-8", errors="ignore") as f:
        for index, sample in enumerate(samples):
            sample_label = samples_label[index]
            tokens, labels = [], []

            for i, token_idx in enumerate(sample):
                tokens.append(vocab.itos[token_idx])
                labels.append(label_vocab.itos[sample_label[i]])


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
                        default=100,
                        type=int)


    args = parser.parse_args()

    print(args.source + "====================================>" + args.target)

    output_path = './LSTM_based/generated_data/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = output_path + args.source + '-' + args.target
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = output_path + '/gen.txt'

    model_path = './LSTM_based/models/'+ args.source + '-' + args.target +'/model.pt'
    params = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    fields = data_utils.load_fields_from_vocab(params["vocab"])
    print(fields.keys())

    model = data_utils.build_test_model(fields, params)

    samples = []
    samples_label = []
    with torch.no_grad():
        for i in tqdm(range(args.generate_number)):
            new_batch, new_label = model.generate(args.max_seq)
    
            samples_label.append(new_label)
            samples.append(new_batch)

    save(samples, samples_label, fields, output_path)


if __name__ == "__main__":
    main()
