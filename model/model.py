import torch
import torch.nn as nn
import torch.nn.functional as F

from .Constants import Constants
class Similarity(nn.Module):
    def __init__(self,mem_dim,hidden_dim,seq_len,num_classes):
        super(Similarity,self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.wh = nn.Linear(2*mem_dim,hidden_dim)
        self.mlp = nn.Linear(seq_len,1,bias=False)
        self.wp = nn.Linear(hidden_dim,num_classes)
    def forward(self,lvec,rvec):
        mult_dist = torch.mul(lvec,rvec)
        abs_dist = torch.abs(torch.add(lvec,-rvec))
        vec_dist = torch.cat((mult_dist,abs_dist),dim=2)
        out = torch.sigmoid(self.wh(self.mlp(vec_dist.transpose(1,2)).squeeze()))
        if out.dim() == 1:
            out = F.log_softmax(self.wp(out),dim=0)
        else:
            out = F.log_softmax(self.wp(out),dim=1)
        return out
class RNNSimilarity(nn.Module):
    def __init__(self,vocab_size,embedding_dim,mem_dim,hid_dim,num_layers,rnn_type,
                 dropout,bidirectional,seq_len,num_classes,sparsity,freeze=True,name=None):
        super(RNNSimilarity,self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.name = name

        # word embedding_dim
        self.word_emb = nn.Embedding(vocab_size,embedding_dim,padding_idx=Constants.PAD,sparse=sparsity)
        if freeze:
            self.word_emb.weight.requires_grad = False
        # RNN layers
        if rnn_type == "RNN":
            self.sent_rnn = nn.RNN(input_size=embedding_dim,hidden_size=mem_dim,num_layers=num_layers,
                                   batch_first=True,dropout=dropout,bidirectional=bidirectional)
        elif rnn_type == "GRU":
            self.sent_rnn = nn.GRU(input_size=embedding_dim, hidden_size=mem_dim, num_layers=num_layers,
                                   batch_first=True, dropout=dropout, bidirectional=bidirectional)
        elif rnn_type == "LSTM":
            self.sent_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=mem_dim, num_layers=num_layers,
                                   batch_first=True, dropout=dropout, bidirectional=bidirectional)
            # (seq_len, batch, num_directions * hidden_size)
        else:
            raise TypeError("Unknown RNN model type:%s"%str(rnn_type))
        hidden_size = mem_dim * 2 if bidirectional else mem_dim * 1
        self.similarity = Similarity(hidden_size,hid_dim,seq_len,num_classes)
    def forward(self,sentA,sentB):
        '''
        :param sentA:(batch,seq_len)
        :param sentB:(batch,seq_len)
        :return:
        '''
        embA = self.word_emb(sentA)
        embB = self.word_emb(sentB)
        hidA,_ = self.sent_rnn(embA)
        hidB,_ = self.sent_rnn(embB)
        out = self.similarity(hidA,hidB)
        return out
