
"""Models module"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import warnings
import data_utils
from encoder import LSTMEncoder
from decoder import LSTMDecoder

warnings.simplefilter("ignore")  # To work with PyTorch 1.2


class LMModel(nn.Module):
    """Define language model class."""

    def __init__(self, fields, args):
        super(LMModel, self).__init__()
        self.vocab, self.label_vocab = fields["sent"].vocab, fields["label"].vocab
        self.vocab_size = len(self.vocab)
        self.label_size = len(self.label_vocab)
        self.unk_idx = self.vocab.stoi[data_utils.UNK_WORD]
        self.padding_idx = self.vocab.stoi[data_utils.PAD_WORD]
        self.bos_idx = self.vocab.stoi[data_utils.BOS_WORD]
        self.eos_idx = self.vocab.stoi[data_utils.EOS_WORD]
        self.device = args.device

        self.embeddings = nn.Embedding(
            self.vocab_size, args.emb_dim, padding_idx=self.padding_idx
        )

        # self.labels=['O', 'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU','PAD']
        self.embed_label=nn.Embedding(self.label_size, args.rnn_size)
        self.fc_label = nn.Linear(args.rnn_size, self.label_size)

        self.encoder = None
        if args.num_z_samples > 0:
            self.encoder = LSTMEncoder(
                hidden_size=args.rnn_size,
                num_layers=args.num_enc_layers,
                bidirectional=args.bidirectional_encoder,
                embeddings=self.embeddings,
                padding_idx=self.padding_idx,
                dropout=args.dropout,
            )
            self.mu = nn.Linear(args.rnn_size, args.z_dim, bias=False)
            self.logvar = nn.Linear(args.rnn_size, args.z_dim, bias=False)
            self.z2h = nn.Linear(args.z_dim, args.rnn_size, bias=False)
            self.z_dim = args.z_dim

        self.decoder = LSTMDecoder(
            hidden_size=args.rnn_size,
            num_layers=args.num_dec_layers,
            embeddings=self.embeddings,
            padding_idx=self.padding_idx,
            unk_idx=self.unk_idx,
            bos_idx=self.bos_idx,
            dropout=args.dropout,
            z_dim=args.z_dim,
            z_cat=args.z_cat,
            inputless=args.inputless,
            word_dropout_rate=args.word_dropout_rate,
        )

        self.dropout = nn.Dropout(args.dropout)
        self.generator = nn.Linear(args.rnn_size, self.vocab_size, bias=False)
        self.num_dec_layers = args.num_dec_layers
        self.rnn_size = args.rnn_size
        self.num_z_samples = args.num_z_samples
        self.use_avg = args.use_avg
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )

        self._init_params(args)


    def _init_params(self, args):
        if args.param_init != 0.0:
            for param in self.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        with torch.no_grad():
            self.embeddings.weight[self.padding_idx].fill_(0)


    def _build_dec_state(self, h):
        h = h.expand(self.num_dec_layers, h.size(0), h.size(1)).contiguous()
        c = h.new_zeros(h.size())
        dec_state = (h, c)
        return dec_state


    def forward(self, text, label):

        text_hidden, _ = self.decoder(text)
        label_hidden = self.embed_label(label)
        fusion_hidden = text_hidden + label_hidden

        text_logits = self.generator(fusion_hidden)
        label_logits = self.fc_label(fusion_hidden)

        return text_logits, label_logits


    def generate(self, max_sent_length):
        text = ['<s>', '[target]']
        label = ['O', 'O']
        text_id = [self.vocab.stoi[t] for t in text]
        label_id = [self.label_vocab.stoi[t] for t in label]

        out_text, out_label = [], []
        
        for t in range(max_sent_length):
            inp_text = torch.tensor(text_id).to(self.device).unsqueeze(1)
            inp_tag = torch.tensor(label_id).to(self.device).unsqueeze(1)

            text_logits, label_logits=self.forward(inp_text,inp_tag)

            # print(label_logits.shape)
            # exit()
            tag = torch.argmax(label_logits, -1)[-1, 0].item()

            text_logits=text_logits[-1, :, :]
            word_weights=text_logits.softmax(-1) # 1,50257
            top_n = word_weights.topk(100)[1] # 1,100
            word_mask = word_weights.scatter(-1, top_n, 1000000001)
            weight_mask = word_mask < 1000000000
            word_weights = word_weights.masked_fill(weight_mask, 0)
            word_indices = torch.multinomial(word_weights, 1)
            word_indices = word_indices.squeeze(1).item()

            label_id.append(tag)
            text_id.append(word_indices)

            if word_indices == self.vocab.stoi['</s>']:
                out_label.append(tag)
                break
            if t != 0:
                out_label.append(tag)
            out_text.append(word_indices)

        # assert len(out_text) == len(out_label), print(len(out_text), len(out_label))

        return out_text, out_label