import torch
import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, embed_size, hidden_size, unigram_num, bigram_num, trigram_num, num_class, dropout):
        super().__init__()
        self.bos_embedding = nn.Embedding(unigram_num, embed_size)
        self.bigram_embedding = nn.Embedding(bigram_num, embed_size)
        self.trigram_embedding = nn.Embedding(trigram_num, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(3 * embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, bos, bigram, trigram):
        # inputs: (b, s)
        x1 = self.bos_embedding(bos)
        # x1: (b, s, ebs)
        x2 = self.bigram_embedding(bigram)
        # x2: (b, s, ebs)
        x3 = self.trigram_embedding(trigram)
        # x3: (b, s, ebs)
        x = torch.cat((x1, x2, x3), dim=-1)
        # x: (b, s, 3 * ebs)
        x = torch.mean(x, dim=1, keepdim=False)
        # x: (b, 3 * ebs)
        x = self.dropout(x)
        x = self.fc1(x)
        # x: (b, h)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x: (b, num)
        return x


