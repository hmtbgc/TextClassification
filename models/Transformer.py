import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        assert d_model % num_head == 0
        self.dk = d_model // num_head
        self.wq = nn.Linear(d_model, self.dk * num_head)
        self.wk = nn.Linear(d_model, self.dk * num_head)
        self.wv = nn.Linear(d_model, self.dk * num_head)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.dk * num_head, d_model)
        self.num_head = num_head

    def forward(self, q, k, v, mask):
        # q: (b, q_len, ebs)
        # k: (b, k_len, ebs)
        # v: (b, v_len, ebs) k_len == v_len
        # mask: (b, 1, q_len, k_len)
        batch_size = q.shape[0]
        Q = self.wq(q).view(batch_size, -1, self.num_head, self.dk)
        K = self.wk(k).view(batch_size, -1, self.num_head, self.dk)
        V = self.wv(v).view(batch_size, -1, self.num_head, self.dk)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2).transpose(2, 3)
        V = V.transpose(1, 2)

        score = torch.matmul(Q, K) / math.sqrt(self.dk)
        mask = mask.repeat(1, self.num_head, 1, 1)
        score.masked_fill_(mask, -1e9)
        attn = torch.softmax(score, dim=3)
        attn = self.dropout(attn)
        x = torch.matmul(attn, V)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_head * self.dk)
        x = self.fc(x)
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (b, s, ebs)
        x = self.fc1(x)
        # x: (b, s, h)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x: (b, s, ebs)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_size, num_head, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_head=num_head, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden_size=hidden_size, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = x + self.dropout(self.norm(self.attn(x, x, x, mask)))
        z = y + self.dropout(self.norm(self.ffn(y)))
        return z

class PositionEncoding(nn.Module):
    def __init__(self, dropout, d_model, device, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, d_model))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.P = self.P.to(device)
        
    def forward(self, x):
        x = x + self.P[:, : x.shape[1], :]
        x = self.dropout(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, embed, d_model, hidden_size, num_head, num_layers, num_class, dropout, device):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embed, freeze=True)
        self.position_encoding = PositionEncoding(dropout=dropout, d_model=d_model, device=device)
        self.encoder = nn.ModuleList([EncoderLayer(d_model=d_model, 
                                    hidden_size=hidden_size,
                                    num_head=num_head,
                                    dropout=dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_class)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, mask):
        # inputs: (b, s)
        # mask: (b, s, s)
        x = self.embedding(inputs)
        # x: (b, s, ebs)
        # x = self.position_encoding(x)
        mask = mask.unsqueeze(1)
        for layer in self.encoder:
            x = layer(x, mask)
        # x: (b, s, ebs)
        x = torch.mean(x, dim=1, keepdim=False)
        # x: (b, ebs)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


         
