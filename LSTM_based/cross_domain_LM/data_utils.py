"""Utilities modules"""
from __future__ import print_function
import io
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torchtext
from model import LMModel
from torchtext.data import Iterator


PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
UNK = 0
BOS_WORD = "<s>"
EOS_WORD = "</s>"

BOS_LABEL = "O"
EOS_LABEL = "O"
PAD_LABEL = "PAD"


def save_fields_to_vocab(fields):
    """Save Vocab objects in Field objects."""
    vocab = []
    for k, f in fields.items():
        if f is not None and "vocab" in f.__dict__:
            vocab.append((k, f.vocab))
    return vocab


def get_model_state_dict(model):
    """Get model state dict."""
    return {k: v for k, v in model.state_dict().items()}


def save_model(fields, model, optimizer, epoch, args):
    """Save model."""
    torch.save(
        {
            "args": args,
            "vocab": save_fields_to_vocab(fields),
            "model": get_model_state_dict(model),
        },
        f"{args.output_dir}/model.pt",
    )


def set_params(args):
    """Set some params."""
    # args.checkpoint_file = "{}.checkpoint".format(args.model_file)

    if args.num_layers != -1:
        args.num_enc_layers = args.num_layers
        args.num_dec_layers = args.num_layers

    if args.num_enc_layers < args.num_dec_layers:
        raise RuntimeError("Expected num_enc_layers >= num_dec_layers")

    if args.num_z_samples == 0:
        args.z_dim = 0
        args.z_cat = False
        args.warmup = 0

    args.beta = 1.0 if args.warmup == 0 else 0.0

    args.device = "cuda" if args.gpuid > -1 else "cpu"


def build_fields():
    """Build fields."""
    fields = {}
    fields["sent"] = torchtext.data.Field(
        init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD
    )
    fields["label"] = torchtext.data.Field(
        init_token=BOS_LABEL, eos_token=EOS_LABEL, pad_token=PAD_LABEL
    )
    return fields


def load_fields_from_vocab(vocab):
    """Load Field objects from Vocab objects."""
    vocab = dict(vocab)
    fields = build_fields()
    for k, v in vocab.items():
        fields[k].vocab = v
    return fields


def build_test_model(fields, params):
    """Build test model."""
    model = LMModel(fields, params["args"])
    model.load_state_dict(params["model"])
    model = model.to(params["args"].device)
    model.eval()
    return model


class LMDataset(torchtext.data.Dataset):
    """Define a dataset class."""

    def __init__(self, fields, filename):
        sents, labels = [], []
        with io.open(filename, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                sentence, label = line.strip().split('####')
                sentence = sentence.lower()
                label = 'O ' + label
                sents.append(sentence)
                labels.append(label)

        fields = [(k, fields[k]) for k in fields]
        examples = []
        for i in range(len(sents)):
            example = torchtext.data.Example.fromlist([sents[i], labels[i]], fields)
            examples.append(example)
        # examples = [torchtext.data.Example.fromlist([sent], fields) for sent in sents]
        super(LMDataset, self).__init__(examples, fields)

    def sort_key(self, ex):
        """Sort by sentence length."""
        return len(ex.sent)


class OrderedIterator(torchtext.data.Iterator):
    """Define an ordered iterator class.
       This class is retrieved from https://github.com/OpenNMT/OpenNMT-py.
       Reference:
       Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart and
       Alexander M. Rush. 2017.  OpenNMT: Open-Source Toolkit for
       Neural Machine Translation. In Proceedings of ACL.
    """

    def create_batches(self):
        """Create batches."""
        if self.train:

            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size,
                        self.batch_size_fn,
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(
                self.data(), self.batch_size, self.batch_size_fn
            ):
                self.batches.append(sorted(b, key=self.sort_key))


def build_dataset_iter(data, args, train=True, shuffle=True):
    """Build dataset iterator."""
    return OrderedIterator(
        dataset=data,
        batch_size=args.batch_size,
        device=args.device,
        train=train,
        shuffle=shuffle,
        repeat=False,
        sort=False,
        sort_within_batch=True,
    )