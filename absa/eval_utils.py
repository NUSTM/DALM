# -*- coding: utf-8 -*-
import os
import numpy as np


def match(pred, gold):
    true_count = 0
    for t in pred:
        if t in gold:
            true_count += 1
    return true_count


def tag2aspect(tag_sequence):
    """
    convert BIO tag sequence to the aspect sequence
    :param tag_sequence: tag sequence in BIO tagging schema
    :return:
    """
    ts_sequence = []
    beg = -1
    for index, ts_tag in enumerate(tag_sequence):
        if ts_tag == 'O':
            if beg != -1:
                ts_sequence.append((beg, index-1))
                beg = -1
        else:
            cur = ts_tag.split('-')[0]  # unified tags
            if cur == 'B':
                if beg != -1:
                    ts_sequence.append((beg, index-1))
                beg = index

    if beg != -1:
        ts_sequence.append((beg, index))
    return ts_sequence


def tag2aspect_sentiment(ts_tag_sequence):
    '''
    support Tag sequence: ['O', 'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU']
    '''
    ts_sequence, sentiments = [], []
    beg, end = -1, -1
    for index, ts_tag in enumerate(ts_tag_sequence):
        if ts_tag == 'O':
            if beg != -1:
                ts_sequence.append((beg, index-1, sentiments[0]))
                beg, sentiments = -1, []
        else:
            cur, pos = ts_tag.split('-')
            if cur == 'B':
                if beg != -1:
                    ts_sequence.append((beg, index-1, sentiments[0]))
                beg, sentiments = index, [pos]
            else:
                if beg != -1:
                    sentiments.append(pos)
    if beg != -1:
        ts_sequence.append((beg, index, sentiments[0]))
    return ts_sequence


def eval_ate_results(pre_list, gold_list):
    assert len(gold_list) == len(pre_list)
    length = len(gold_list)

    TP, FN, FP = 0, 0, 0

    for i in range(length):
        gold = gold_list[i]
        pred = pre_list[i]
        assert len(gold) == len(pred)
        gold_aspects = tag2aspect(gold)
        pred_aspects = tag2aspect(pred)
        n_hit = match(pred=pred_aspects, gold=gold_aspects)
        TP += n_hit
        FP += (len(pred_aspects) - n_hit)
        FN += (len(gold_aspects) - n_hit)
    precision = float(TP) / float(TP + FP + 0.00001)
    recall = float(TP) / float(TP + FN + 0.0001)
    F1 = 2 * precision * recall / (precision + recall + 0.00001)
    return precision, recall, F1


def eval_absa_results(pre_list, gold_list):
    assert len(gold_list) == len(pre_list)
    length = len(gold_list)

    TP_A, FN_A, FP_A = 0, 0, 0

    for i in range(length):
        gold = gold_list[i]
        pred = pre_list[i]
        assert len(gold) == len(pred)
        gold_aspects = tag2aspect_sentiment(gold)
        pred_aspects = tag2aspect_sentiment(pred)
        n_hit_a = match(pred=pred_aspects, gold=gold_aspects)
        TP_A += n_hit_a
        FP_A += (len(pred_aspects) - n_hit_a)
        FN_A += (len(gold_aspects) - n_hit_a)
    
    # print(TP_A, FP_A, FN_A)

    precision_a = float(TP_A) / float(TP_A + FP_A + 0.00001)
    recall_a = float(TP_A) / float(TP_A + FN_A + 0.00001)
    F1_a = 2 * precision_a * recall_a / (precision_a + recall_a + 0.00001)

    return precision_a, recall_a, F1_a
