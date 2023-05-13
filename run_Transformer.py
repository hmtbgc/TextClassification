import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import logging
from models.Transformer import Transformer


model_name="Transformer"

def init_logging(path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

logger = init_logging(f"./log/{model_name}.txt")

pretrained_embedding_path = "./data/pretrained_wordvector/sgns.sogou.char"
embed = []
word2idx = dict()
idx2word = dict()

size = None
with open(pretrained_embedding_path, "r") as f:
    idx = 0
    f.readline()
    for line in tqdm(f):
        x = line.strip().split(' ')
        word = x[0]
        vector = np.asarray(x[1:], dtype=np.float32)
        size = vector.shape
        embed.append(vector)
        word2idx[word] = idx
        idx2word[idx] = word
        idx += 1


avg = sum(embed) / len(embed)
embed.append(avg)
word2idx['<UNK>'] = idx
idx2word[idx] = '<UNK>'
idx += 1

embed.append(np.random.normal(size=size))
word2idx['<PAD>'] = idx
idx2word[idx] = '<PAD>'
idx += 1

embed = torch.from_numpy(np.array(embed)).float()

class MyDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r") as f:
            for line in tqdm(f):
                x = line.split('\t')
                sen, label = x[0], x[1]
                self.data.append((sen, int(label)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
                
    def collate(self, batchs):
        sen_out = []
        tot_sen = [pair[0] for pair in batchs]
        tot_label = [pair[1] for pair in batchs]
        max_len = max([len(sen) for sen in tot_sen])
        for sen in tot_sen:
            temp = []
            for ch in sen:
                if (ch not in word2idx):
                    temp.append(word2idx['<UNK>'])
                else:
                    temp.append(word2idx[ch])
            temp += [word2idx['<PAD>']] * (max_len - len(sen))
            sen_out.append(temp)
        
        return torch.from_numpy(np.array(sen_out)), torch.from_numpy(np.array(tot_label))
    
train_dataset = MyDataset("./data/train.txt")
test_dataset = MyDataset("./data/test.txt")
valid_dataset = MyDataset("./data/valid.txt")

batch_size = 256

train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=valid_dataset.collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(embed=embed, d_model=300, hidden_size=600, num_layers=4, num_head=6, num_class=14, dropout=0.5, device=device)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

eval_every = 2

PAD_ID = word2idx['<PAD>']

def get_mask(data):
    # data: (b, s)
    batch_size, s = data.shape[0], data.shape[1]
    pad_attn_mask = data.data.eq(PAD_ID).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.repeat(1, s, 1)
    # pad_attn_mask.shape: (b, s, s)
    return pad_attn_mask

def eval(model, dataloader, device):
    model.eval()
    acc = 0
    tot = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data, label = batch
            data = data.to(device)
            mask = get_mask(data).to(device)
            out = model(data, mask)
            predicted = torch.argmax(out, dim=-1)
            predicted = predicted.cpu()
            acc += (predicted == label).sum()
            tot += predicted.shape[0]
        return acc / tot

Epoch = 100
best_acc = 0.0
best_epoch = -1
early_stop = 5
for epoch in range(Epoch):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(train_dataloder):
        data, label = batch
        data = data.to(device)
        mask = get_mask(data).to(device)
        label = label.to(device)
        predicted = model(data, mask)
        loss = loss_fn(predicted, label)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info(f"epoch:{epoch}, loss:{epoch_loss / len(train_dataloder):.4f}")
    if (epoch % eval_every == 0):
        acc = eval(model, valid_dataloader, device)
        if (acc > best_acc):
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), f"./pt/{model_name}.pt")
        else:
            early_stop -= 1
        logger.info(f"epoch:{epoch}, valid acc: {acc * 100:.4f}%, best valid acc: {best_acc * 100:.4f}, at epoch {best_epoch}")
        if (early_stop == 0):
            logger.info(f"early stop!")
            break
            
test_model = Transformer(embed=embed, d_model=300, hidden_size=600, num_layers=4, num_head=6, num_class=14, dropout=0.5, device=device)
test_model.load_state_dict(torch.load(f"./pt/{model_name}.pt"))
test_model = test_model.to(device)

test_model.eval()
with torch.no_grad():
    acc = eval(test_model, test_dataloader, device)
    logger.info(f"test acc: {acc * 100:.4f}%")

