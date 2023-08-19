import numpy as np
import json
import os
import copy
from collections import defaultdict, namedtuple
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,  text_a, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, label_id):
        self.input_ids = input_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)

    def read_txt(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fp:
            text = fp.readlines()
        lines = {}
        id = 0
        for _, t in enumerate(text):
            sentence, label = t.split('####')  # this is a example####O O O O
            sentence = sentence.lower().split()
            label = label.split()

            assert len(label) == len(sentence), print(len(label),len(sentence)) 
            lines[id] = {'sentence': ' '.join(sentence), 'label': label}
            id += 1
        print('normal text and label:')
        print(lines[0])
        print(lines[1])
        return lines


class ABSAProcessor(DataProcessor):
    """Processor for the SemEval Aspect Extraction and end2end absa ."""

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self.read_txt(data_dir)

        return self._create_examples(
            lines)

    def get_labels(self):
        """See base class."""
        labels = ['O', 'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU','PAD']
        return labels

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""

        examples = []
        ids = 0

        for i in range(len(lines)):
            text_a = lines[i]['sentence']
            label =  lines[i]['label']
            examples.append(
                InputExample(text_a=text_a, label=label))
            ids += 1
        return examples


def tl2gpttl(text, label, tokenizer, label_list, max_seq):
    label_dict = {}
    for (i, labe) in enumerate(label_list):
        label_dict[labe] = i
    pre_label = 0
    cur_pos = 0
    gpt_label = []

    text_list = text.split()
    type = text_list[0]
    text = ' '.join(text_list[1:])
    label = label[1:]

    gpt_text = tokenizer.tokenize(text)

    for i, sub_token in enumerate(gpt_text):
        if i != 0 and sub_token[0] != 'Ġ':
            if pre_label == 'O':
                gpt_label.append('O')
            else:
                gpt_label.append('I' + pre_label[1:])
        else:
            gpt_label.append(label[cur_pos])
            pre_label = label[cur_pos]
            cur_pos += 1

    if type == '[target]':
        gpt_text = ['<|endoftext|>', 'Á'] + gpt_text + ['<|endoftext|>']
    elif type == '[source]':
        gpt_text = ['<|endoftext|>', 'Ã'] + gpt_text + ['<|endoftext|>']
    gpt_label = ['O', 'O'] + gpt_label + ['O']

    while len(gpt_text) < max_seq:
        gpt_text.append('Å')
        gpt_label.append('PAD')

    if len(gpt_text) > max_seq:
        gpt_text = gpt_text[:max_seq-1]+ ['<|endoftext|>']
        gpt_label = gpt_label[:max_seq-1]+['O']

    gpt_text = tokenizer.convert_tokens_to_ids(gpt_text)
    assert len(gpt_text) == len(gpt_label), print(gpt_label)
    
    gpt_label=['O']+gpt_label
    gpt_label = [label_dict[i] for i in gpt_label]

    return gpt_text, gpt_label  #input_text :-1 out_put_text 1:  input_label=[:-2]  output_label[1:-1]


def convert_examples_to_features(examples, label_list, max_seq, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):

        input_ids,label_id=tl2gpttl(example.text_a,example.label,tokenizer,label_list,max_seq)
        features.append(
            InputFeatures(
                input_ids=input_ids,
                label_id=label_id))
    
    return features

