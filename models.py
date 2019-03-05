"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()

        # print("vectors: ", word_vectors)
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors = char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.self_att = layers.SelfAttention(hidden_size = 8 * hidden_size, 
                                                drop_prob = drop_prob)



        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

        self.batch_size = 64
        self.hidden_size = hidden_size

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        # print("masks: ", c_mask, q_mask)
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        # print(type(c_len), c_len.size(), q_len.size())
        # print("c_len: ", c_len)
        # print("c_lens size", c_len.size())
        # print("q_len size: ", q_len.size())


        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        # c_emb = c_emb.view(64, 369, 100)
        # q_emb = q_emb.view(64, )
        # print(c_emb.size())
        # print("here")
        # print("c_emb size: ", c_emb.size())
        # print("q_emb size: ", q_emb.size())

        # c_len_ = int(c_emb.size()[0]/self.batch_size)
        # q_len_ = int(q_emb.size()[0]/self.batch_size)

        # c_len_new = cw_idxs.size()[1]
        # print("new c_len: ", c_len_new)
        # print("other? : ", cc_idxs.size()[1])
        # print("old: ", c_len_)

        # print(c_len_, q_len_)

        c_len_ = cw_idxs.size()[1]
        q_len_ = qw_idxs.size()[1]

        assert (c_len_ <= 400)
        assert (q_len_ <= 50)

        # print(c_len_, q_len_)

        # c_emb = c_emb.view(self.batch_size, c_len_, self.hidden_size)
        # q_emb = q_emb.view(self.batch_size, q_len_, self.hidden_size)

        c_emb = c_emb.view(-1, c_len_, self.hidden_size)
        q_emb = q_emb.view(-1, q_len_, self.hidden_size)

        print("c_emb size: ", c_emb.size())

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        # assert (c_enc.size()[0] == self.batch_size)
        assert (c_enc.size()[1] == c_len_)
        assert (c_enc.size()[2] == 2 * self.hidden_size)

        # assert (q_enc.size()[0] == self.batch_size)
        assert (q_enc.size()[1] == q_len_)
        assert (q_enc.size()[2] == 2 * self.hidden_size)


        # print("hre 2")
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        
        # assert (att.size()[0] == self.batch_size)
        assert (att.size()[1] == c_len_)
        assert(att.size()[2] == 8 * self.hidden_size)




        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        # assert (mod.size()[0] == self.batch_size)
        assert (mod.size()[1] == c_len_)
        assert (mod.size()[2] == 2 * self.hidden_size)

        # print("here 3")
        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
