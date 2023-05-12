import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import logging
from collections import Counter
from models.FastText import FastText

model_name="FastText"

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

# generate n-gram
def get_n_gram(path):
    tot_1_gram = []
    tot_2_gram = []
    tot_3_gram = []
    with open(path, "r") as f:
        for line in tqdm(f):
            x = line.split('\t')
            sen = x[0]
            temp = [word2idx[ch] if ch in word2idx else word2idx['<UNK>'] for ch in sen]
            tot_1_gram += temp
            tot_2_gram += [(temp[i], temp[i+1]) for i in range(len(temp)-1)]
            tot_3_gram += [(temp[i], temp[i+1], temp[i+2]) for i in range(len(temp)-2)]
    return tot_1_gram, tot_2_gram, tot_3_gram

tot_1_gram = []
tot_2_gram = []
tot_3_gram = []
for path in ["./data/train.txt", "./data/valid.txt", "./data/test.txt"]:
    x, y, z = get_n_gram(path)
    tot_1_gram += x
    tot_2_gram += y
    tot_3_gram += z


def topk(lst, k):
    counter = Counter(lst)
    return counter.most_common(k)

k = 10000
tot_1_gram_topk = topk(tot_1_gram, k)
tot_2_gram_topk = topk(tot_2_gram, k)
tot_3_gram_topk = topk(tot_3_gram, k)


unigram2idx = dict()
bigram2idx = dict()
trigram2idx = dict()

for i, item in tqdm(enumerate(tot_1_gram_topk)):
    unigram2idx[item[0]] = i

for i, item in tqdm(enumerate(tot_2_gram_topk)):
    bigram2idx[item[0]] = i

for i, item in tqdm(enumerate(tot_3_gram_topk)):
    trigram2idx[item[0]] = i

unigram2idx['<UNK>'] = len(unigram2idx)
unigram2idx['<PAD>'] = len(unigram2idx)
bigram2idx['<UNK>'] = len(bigram2idx)
bigram2idx['<PAD>'] = len(bigram2idx)
trigram2idx['<UNK>'] = len(trigram2idx)
trigram2idx['<PAD>'] = len(trigram2idx)

class MyDataset(Dataset):
    def __init__(self, path, ngram=False):
        self.data = []
        with open(path, "r") as f:
            for line in tqdm(f):
                x = line.split('\t')
                sen, label = x[0], x[1]
                self.data.append((sen, int(label)))
        self.ngram = ngram
    
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
        
        PAD_ID = word2idx['<PAD>']

        if (self.ngram):
            unigram = []
            for sen in sen_out:
                temp = []
                for i in range(len(sen)):
                    pair = sen[i]
                    if (pair == PAD_ID):
                        temp.append(unigram2idx['<PAD>'])
                    elif (pair not in unigram2idx):
                        temp.append(unigram2idx['<UNK>'])
                    else:
                        temp.append(unigram2idx[pair])
                unigram.append(temp)
            
            bigram = []
            for sen in sen_out:
                temp = []
                for i in range(len(sen)):
                    if (i == len(sen) - 1):
                        pair = (sen[i], PAD_ID)
                    else:
                        pair = (sen[i], sen[i+1])
                    if (pair == (PAD_ID, PAD_ID)):
                        temp.append(bigram2idx['<PAD>']) 
                    elif (pair not in bigram2idx):
                        temp.append(bigram2idx['<UNK>'])
                    else:
                        temp.append(bigram2idx[pair])
                bigram.append(temp)

            trigram = []
            for sen in sen_out:
                temp = []
                for i in range(len(sen)):
                    if (i == len(sen) - 1):
                        pair = (sen[i], PAD_ID, PAD_ID)
                    elif (i == len(sen) - 2):
                        pair = (sen[i], sen[i+1], PAD_ID)
                    else:
                        pair = (sen[i], sen[i+1], sen[i+2])
                    if (pair == (PAD_ID, PAD_ID, PAD_ID)):
                        temp.append(trigram2idx['<PAD>'])
                    elif (pair not in trigram2idx):
                        temp.append(trigram2idx['<UNK>'])
                    else:
                        temp.append(trigram2idx[pair])
                trigram.append(temp)

            return torch.from_numpy(np.array(unigram)), torch.from_numpy(np.array(bigram)), \
                    torch.from_numpy(np.array(trigram)), torch.from_numpy(np.array(tot_label))
        
        return torch.from_numpy(np.array(sen_out)), torch.from_numpy(np.array(tot_label))
    
train_dataset = MyDataset("./data/train.txt", ngram=True)
test_dataset = MyDataset("./data/test.txt", ngram=True)
valid_dataset = MyDataset("./data/valid.txt", ngram=True)

batch_size = 256

train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=valid_dataset.collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FastText(embed_size=100, hidden_size=128, unigram_num=len(unigram2idx), \
                 bigram_num=len(bigram2idx), trigram_num=len(trigram2idx), num_class=14, dropout=0.5)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

eval_every = 2

def eval(model, dataloader, device):
    model.eval()
    acc = 0
    tot = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            bos_data, bigram_data, trigram_data, label = batch
            bos_data = bos_data.to(device)
            bigram_data = bigram_data.to(device)
            trigram_data = trigram_data.to(device)
            out = model(bos_data, bigram_data, trigram_data)
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
        bos_data, bigram_data, trigram_data, label = batch
        label = label.to(device)
        bos_data = bos_data.to(device)
        bigram_data = bigram_data.to(device)
        trigram_data = trigram_data.to(device)
        predicted = model(bos_data, bigram_data, trigram_data)
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

test_model = FastText(embed_size=100, hidden_size=128, unigram_num=len(unigram2idx), \
                 bigram_num=len(bigram2idx), trigram_num=len(trigram2idx), num_class=14, dropout=0.5)
test_model.load_state_dict(torch.load(f"./pt/{model_name}.pt"))
test_model = test_model.to(device)

test_model.eval()
with torch.no_grad():
    acc = eval(test_model, test_dataloader, device)
    logger.info(f"test acc: {acc * 100:.4f}%")
