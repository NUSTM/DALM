# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import io
import torch
from torch.utils.data import Dataset


def get_labels(task = 'absa'):
    '''
    Get label list for the ABSA task
    return: label_list
    '''
    labels = ['O', 'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU']
    return labels


def is_clean_tag(tag_list):
    pre_tag = 'O'
    for tag in tag_list:
        if tag != 'O':
            if tag[:2] not in {'B-', 'I-'}: return False
            if pre_tag != 'O' and tag[2:] != pre_tag[2:]: return False

        pre_tag = tag
    return True


def is_clean_tok(tok_list):
    found = False
    for tok in tok_list:
        if tok != "<unk>":
            found = True
        if tok[:2] in {"B-", "I-"}:
            return False
    if not found:
        return False
    return True


def read_absa_raw_data(data_path):
    '''
    Read raw train/test data for the ABSA task
    input: data_path
    return: sentence_lists, label_lists
        sentence_lists: [['the', 'food', 'is', 'great', '.'], ...]
        label_lists: [['O', 'B-POS', 'O', 'O', 'O'], ...]
    '''
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                _, t = line.split('####')
                sentence, label = [], []
                for i in t.split():
                    if len(i.split('=')) > 2:
                        sentence.append('='.join(i.split('=')[:-1]))
                        label.append(i.split('=')[-1])
                    else:
                        if len(i.split('=')[0]) > 0:
                            sentence.append(i.split('=')[0])
                            label.append(i.split('=')[-1])

                cur_label = []
                pre = 'O'
                for tag in label:
                    if tag != 'O':
                        if (pre != 'O'):
                            cur_label.append('I' + tag[1:])
                        else:
                            cur_label.append('B' + tag[1:])
                    else:
                        cur_label.append("O")
                    pre = tag

                assert len(cur_label) == len(sentence), print(sentence, cur_label)
                sents.append(sentence)
                labels.append(cur_label)
    return sents, labels


def read_generated_data(data_path):
    '''
    Read generated data
    input: data_path
    return: sentence_lists, label_lists
        sentence_lists: [['the', 'food', 'is', 'great', '.'], ...]
        label_lists: [['O', 'B-POS', 'O', 'O', 'O'], ...]
    '''
    data = []
    with io.open(data_path, encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            data.append(line.strip())

    data = list(dict.fromkeys(data))

    sents, labels = [], []
    for line in data:
        if line.count('#')>8:
            continue
        tokens,tags=line.split('####')

        flag = True
        for line2 in data:
            if line2.count('#')>8:
                continue
            tokens2,tags2 = line2.split('####')
            if tokens == tokens2 and tags != tags2:
                flag = False
                break
        if not flag:
            continue

        text=tokens.split()
        tag=tags.split()
            
        if not is_clean_tok(text):
            continue

        if not is_clean_tag(tag):
            continue

        if(len(text)!=len(tag)):
            continue

        # for i in range(len(tag)):
        #     if tag[i] in {'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU'}:
        #         tag[i]='T-'+tag[i][2:]

        sents.append(text)
        labels.append(tag)
    return sents, labels


def read_filtered_data(data_path):
    '''
    读取生成的absa样本
    input: data_path
    return: sentence_lists, label_lists
        sentence_lists: [['the', 'food', 'is', 'great', '.'], ...]
        label_lists: [['O', 'B-POS', 'O', 'O', 'O'], ...]
    '''
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                sentence, label = line.split('####')[0].strip().split(), line.split('####')[1].strip().split()
                sents.append(sentence)
                # labels.append(label)

                cur_label = []
                pre = 'O'
                for tag in label:
                    if tag != 'O':
                        if (pre != 'O'):
                            cur_label.append('I' + tag[1:])
                        else:
                            cur_label.append('B' + tag[1:])
                    else:
                        cur_label.append("O")
                    pre = tag

                labels.append(cur_label)
    return sents, labels


def read_dp_data(data_path):
    '''
    Read DP data
    input: data_path
    return: sentence_lists, label_lists
        sentence_lists: [['the', 'food', 'is', 'great', '.'], ...]
        label_lists: [['O', 'B-POS', 'O', 'O', 'O'], ...]
    '''
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                sentence, label = line.split('####')[0].strip().split(), line.split('####')[1].strip().split()
                sents.append(sentence)
                labels.append(label)

    for id in range(len(labels)):
        new_label = []
        for t in labels[id]:
            if 'OP' in t:
                t ='O'
            new_label.append(t)
        labels[id] = new_label

    return sents, labels


def sub_tokenize(tokenizer, words, labels):
    '''
    input: tokenizer, word_list, label_list
    return: token_list, new_label_list
    '''
    output_tokens, output_labels = [], []

    for idx, token in enumerate(words):
        sub_tokens = tokenizer.tokenize(token)
        for jdx, sub_token in enumerate(sub_tokens):
            output_tokens.append(sub_token)

            if labels[idx].startswith('B') and jdx != 0:
                output_labels.append(labels[idx].replace('B', 'I'))
            else:
                output_labels.append(labels[idx])
    
    return output_tokens, output_labels


def get_bert_input(sentence, label, tokenizer, max_len, label_map):
    '''
    input: word_list, label_list, tokenizer, max_len, label_map
    return tensor(input_id), tensor(input_mask), tensor(label_id)
    '''
    sub_token, sub_label = sub_tokenize(tokenizer, [word.lower() for word in sentence], label)

    # Account for [CLS] and [SEP] with "- 2"
    if len(sub_token) > max_len - 2:
        sub_token = sub_token[0:(max_len - 2)]
        sub_label = sub_label[0:(max_len - 2)]

    assert len(sub_token) == len(sub_label)

    tokens = []
    tokens.append("[CLS]")
    for token in sub_token:
        tokens.append(token)

    tokens.append("[SEP]")

    input_id = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_id)

    # Zero-pad up to the sequence length.
    while len(input_id) < max_len:
        input_id.append(0)
        input_mask.append(0)

    assert len(input_id) == max_len
    assert len(input_mask) == max_len

    # -1 is the index to ignore use 0
    label_id = [-1] * len(input_id)
    # truncate the label length if it exceeds the limit.
    lb = [label_map[label] for label in sub_label]
    if len(lb) > max_len - 2:
        lb = lb[0:(max_len - 2)]
    # print(sum(lb))
    label_id[1:len(lb) + 1] = lb  # 前后都是-1

    return torch.tensor(input_id, dtype=torch.long), torch.tensor(input_mask, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)


def convert_raw_data(token_list, label_list):
    '''
    input: token_list, label_list
        token_list: ['i', 'was', 'please', '##ntly', 'surprised', 'at', 'the' ,'taste', '.']
        label_list: ['O', 'O', 'O' ,'O' ,'O' , 'O', 'O', 'T-POS', 'O']
    return: word_list, new_label_list
        word_list: ['i', 'was', 'pleasently', 'surprised', 'at', 'the' ,'taste', '.']
        new_label_list: ['O', 'O', 'O' ,'O' , 'O', 'O', 'T-POS', 'O']
    '''
    new_label_list = []
    for tag in label_list:
        if tag != 'O': new_label_list.append('T-' + tag[2:])
        else: new_label_list.append('O')

    text, label = [], []
    for i in range(len(token_list)):
        w, t = token_list[i], new_label_list[i]

        if t != 'O' : t = 'T-' + t[2:]

        if '##' in w:
            if (text == []):
                label.append(t)
                text.append(w[2:])
            else:
                text[-1] = text[-1] + w[2:]
        else:
            label.append(t)
            text.append(w)

    assert len(text) == len(label), print(len(text), len(label))

    return text, label


def unify_data_num(source_dataset, target_dataset):
    '''
    Unify num of source and target data, and ensure len(source_dataset) == len(target_dataset)
    input: source_dataset, target_dataset
    return: source_dataset, target_dataset
    '''
    source_train_num, target_train_num = len(source_dataset), len(target_dataset)
    if source_train_num < target_train_num:
        for i in range(source_train_num, target_train_num):
            source_dataset.input_ids.append(source_dataset.input_ids[i % source_train_num])
            source_dataset.input_masks.append(source_dataset.input_masks[i % source_train_num])
            source_dataset.label_ids.append(source_dataset.label_ids[i % source_train_num])
    else:
        for i in range(target_train_num, source_train_num):
            target_dataset.input_ids.append(target_dataset.input_ids[i % target_train_num])
            target_dataset.input_masks.append(target_dataset.input_masks[i % target_train_num])
            target_dataset.label_ids.append(target_dataset.label_ids[i % target_train_num])

    assert len(source_dataset) == len(target_dataset), print(len(source_dataset), len(target_dataset))

    return source_dataset, target_dataset


class OurDataset(Dataset):
    def __init__(self, data_path, task, tokenizer, max_len=128):
        self.data_path = data_path
        self.task = task
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.input_ids = []
        self.input_masks = []
        self.label_ids = []

        self._build_examples()
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_id = self.input_ids[index].squeeze()
        input_mask = self.input_masks[index].squeeze()
        label_id = self.label_ids[index].squeeze()

        return {"input_id": input_id, "input_mask": input_mask, "label_id": label_id}

    def _build_examples(self):
        label_list = get_labels(self.task)
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        if "raw_data" in self.data_path:
            sentences, labels = read_absa_raw_data(self.data_path)
        elif "dp" in self.data_path:
            sentences, labels = read_dp_data(self.data_path)
        elif "filter" in self.data_path:
            sentences, labels = read_filtered_data(self.data_path)
        else:
            sentences, labels = read_generated_data(self.data_path)

        # for i in range(5):
        #     print(sentences[i])
        #     print(labels[i])

        for i in range(len(sentences)):
            sentence, label = sentences[i], labels[i]

            input_id, input_mask, label_id = get_bert_input(sentence, label, self.tokenizer, self.max_len, label_map)

            self.input_ids.append(input_id)
            self.input_masks.append(input_mask)
            self.label_ids.append(label_id)
