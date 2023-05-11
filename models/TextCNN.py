import torch.nn as nn
import torch.nn.functional as F
import torch

class TextCNN(nn.Module):
    def __init__(self, out_channels, embed, embed_size, num_class, dropout):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embed, freeze=True)
        self.kernel1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(2, embed_size))
        self.kernel2 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(3, embed_size))
        self.kernel3 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(4, embed_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(3 * out_channels, num_class)
        
    def forward(self, inputs):
        # inputs: (b, s)
        x = self.embedding(inputs)
        # (b, s, ebs)
        x = x.unsqueeze(1)
        # (b, 1, s, ebs)
        out1 = self.kernel1(x)
        out2 = self.kernel2(x)
        out3 = self.kernel3(x)
        # (b, out, s-1, 1)
        # (b, out, s-2, 1)
        # (b, out, s-3, 1)
        out1 = torch.squeeze(out1, dim=-1)
        out2 = torch.squeeze(out2, dim=-1)
        out3 = torch.squeeze(out3, dim=-1)
        # (b, out, s-1)
        # (b, out, s-2)
        # (b, out, s-3)
        out1 = F.max_pool1d(out1, out1.shape[-1])
        out2 = F.max_pool1d(out2, out2.shape[-1])
        out3 = F.max_pool1d(out3, out3.shape[-1])
        # (b, out, 1)
        # (b, out, 1)
        # (b, out, 1)
        out1 = torch.squeeze(out1, dim=-1)
        out2 = torch.squeeze(out2, dim=-1)
        out3 = torch.squeeze(out3, dim=-1)
        # (b, out)
        # (b, out)
        # (b, out)
        out = torch.cat([out1, out2, out3], dim=-1)
        # (b, 3 * out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out        