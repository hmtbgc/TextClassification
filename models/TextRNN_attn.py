import torch
import torch.nn.functional as F
import torch.nn as nn

class TextRNN_attn(nn.Module):
    def __init__(self, embed, embed_size, hidden_size, num_class, dropout):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embed, freeze=True)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Linear(2 * hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, num_class)
        
    def forward(self, inputs):
        # inputs: (b, s)
        x = self.embedding(inputs)
        # (b, s, ebs)
        out, _ = self.lstm(x)
        # (b, s, 2h)
        attn = self.attn(out)
        # (b, s, 1)
        attn = F.softmax(attn, dim=1)
        # (b, s, 1)
        out = torch.sum(attn * out, dim=1, keepdim=False)
        # (b, 2h)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out
        
        