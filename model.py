import torch
import torch.nn as nn

import torch.nn.functional as F

d_model = 300
d_hidden = 150
p_drop = 0.2

class HARM_Model(nn.Module):
    def __init__(self, n_vocab, glove_emb=None):
        super().__init__()
        if glove_emb is None:
            self.embedding = nn.Embedding(n_vocab, d_model, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(glove_emb, freeze=False, padding_idx=0)
        # encoders
        self.enc_q = nn.GRU(
            input_size=d_model, hidden_size=d_hidden, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.enc_d = nn.GRU(
            input_size=d_model, hidden_size=d_hidden, num_layers=1,
            batch_first=True, bidirectional=True
        )
        # weights for cross attention
        self.Wc = nn.Linear(3 * d_model, 1, bias=False)

        v = 1 / (d_model ** 2)
        # weights for self-attention
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.bq = nn.Parameter(torch.empty((d_model,)).uniform_(-v, v))

        self.Wd1 = nn.Linear(4 * d_model, d_model, bias=False)
        self.bd1 = nn.Parameter(torch.empty((d_model,)).uniform_(-v, v))
        
        self.Wd2 = nn.Linear(4 * d_model, d_model, bias=False)
        self.bd2 = nn.Parameter(torch.empty((d_model,)).uniform_(-v, v))

        # ffn
        self.rescale = nn.Linear(4 * d_model, d_model)
        self.ffn = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(p=p_drop)
        self.tanh = nn.Tanh()

    # ud: B,M,H;    uq: B,N,H
    def cross_attn(self, uq, ud):
        ud_ = ud.unsqueeze(2).expand(-1, -1, uq.size(1), -1) # B,M,N,H
        uq_ = uq.unsqueeze(1).expand(-1, ud.size(1), -1, -1) # B,M,N,H
        matrix = self.Wc(torch.cat([ud_, uq_, ud_ * uq_], dim=-1)).squeeze(-1) # B,M,N
        Sd2q, Sq2d = F.softmax(matrix, dim=2), F.softmax(matrix, dim=1) # B,M,N
        Ad2q = Sd2q.bmm(uq) # B,M,H
        Aq2d = Sd2q.bmm(Sq2d.transpose(1, 2)).bmm(ud) # B,M,H
        vd = torch.cat([ud, Ad2q, ud * Ad2q, ud * Aq2d], dim=-1) # B,M,4H
        return vd
    
    # value: B,N,H; W: Linear(H->h); b: tensor(h)
    def self_attn(self, value, W, b):
        weights = self.tanh(W(value)).matmul(b) # B,N
        norm_weights = F.softmax(weights, -1)
        return norm_weights.unsqueeze(1).bmm(value) # B,1,H

    # query: B,L; segments: [B,Ni,...]
    def forward(self, query, segments):
        # embedding lookup layer
        query = self.dropout(self.embedding(query))
        segments = [self.dropout(self.embedding(seg)) for seg in segments]
        # encoder layer
        uq = self.dropout(self.enc_q(query)[0])
        ud_list = [self.dropout(self.enc_d(seg)[0]) for seg in segments]
        # cross attention layer
        vd_list = [self.dropout(self.cross_attn(uq, ud)) for ud in ud_list]
        # self word attention layer
        x_q = self.dropout(self.self_attn(uq, self.Wq, self.bq)) # B,1,H
        x_d = torch.cat([self.dropout(self.self_attn(vd, self.Wd1, self.bd1)) for vd in vd_list], dim=1)
        # self sentence layer
        y_d = self.dropout(self.self_attn(x_d, self.Wd2, self.bd2)) # B,1,4H
        # ffn
        y_d_ = self.rescale(y_d)
        r = self.ffn(y_d_ * x_q).reshape(-1) # B,1,1 -> B
        return r

if __name__ == "__main__":
    import random, json
    from data_util import TrainLoader

    with open("dump/vocab.json", 'r') as f:
        vocab = json.load(f)
    glove_emb = torch.load("dump/glove.emb")

    harm = HARM_Model(10000, glove_emb)
    ld = TrainLoader("data/train.csv", vocab, "cpu")

    for q, docs in ld():
        r = harm(q, docs)
        print(r)