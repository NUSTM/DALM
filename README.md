# Cross-Domain Data Augmentation with Domain-Adaptive Language Modeling for Aspect-Based Sentiment Analysis

The code for our ACL2023 paper (https://aclanthology.org/2023.acl-long.81/)

Jianfei Yu, Qiankun Zhao, Rui Xia. "Cross-Domain Data Augmentation with Domain-Adaptive Language Modeling for Aspect-Based Sentiment Analysis"

## Datasets

The training data comes from four domains: Restaurant(R) 、 Laptop(L) 、 Service(S) 、 Devices(D).  

The in-domain corpus(used for training BERT-E) come from [yelp](https://www.yelp.com/dataset/challenge) and [amazon reviews](http://jmcauley.ucsd.edu/data/amazon/links.html). 

Click here to get [BERT-E](https://pan.baidu.com/s/1hNyNCyfOHzznuPbxT1LNFQ) (BERT-Extented) , and the extraction code is by0i. (Please specify the directory where BERT is stored in modelconfig.py.)

## Usage

### 1. Domain-Adaptive Pseudo Labeling

To assign pseudo labels to unlabeled data in the target domain, run below code:
```
bash pseudo_label.sh
```

### 2. Domain-Adaptive Language Modeling

Train a domain-adaptive language model, generate target-domain labeled data, and finally use the generated data for the main tasks.
We use LSTM and GPT2 as decoder in language modeling respectively.

2.1 To train the GPT2-based DALM for data generation and evaluation, run below code:
```
bash GPT2.sh
```

2.2 To train the LSTM-based DALM for data generation and evaluation, run below code:
```
bash LSTM.sh
```

## Acknowledgements

- Some code in LSTM-based language modeling are based on the codes of [DAGA](https://aclanthology.org/2020.emnlp-main.488/), many thanks!

