import torch
import torch.nn as nn
import torch.nn.functional as F

class BagOfWord(nn.Module):
    def __init__(self, embed, embed_size, hidden_size, num_class, dropout):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embed, freeze=True)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, inputs):
        # inputs: (b, s)
        x = self.embedding(inputs)
        # x: (b, s, ebs)
        x = self.fc1(x)
        # x: (b, s, h)
        x = torch.mean(x, dim=1, keepdim=False)
        # x: (b, h)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x: (b, num)
        return x
