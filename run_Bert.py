import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import logging
from transformers import BertTokenizer, BertModel

model_name="Bert"

def init_logging(path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

logger = init_logging(f"./log/{model_name}.txt")

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-chinese",
    cache_dir="./cache/bert-base-chinese/"
)

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
        sens = [i[0] for i in batchs]
        labels = [i[1] for i in batchs]
        
        batch_encode = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens,
                                                   add_special_tokens=True,
                                                   truncation=True,
                                                   padding="max_length",
                                                   max_length=32,
                                                   return_tensors="pt",
                                                   return_length=True)
        input_ids = batch_encode["input_ids"]
        attention_mask = batch_encode["attention_mask"]
        
        return input_ids, attention_mask, torch.LongTensor(labels) 
    
train_dataset = MyDataset("./data/train.txt")
test_dataset = MyDataset("./data/test.txt")
valid_dataset = MyDataset("./data/valid.txt")

batch_size = 256

train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=valid_dataset.collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
class Model(nn.Module):
    def __init__(self, hidden_size, num_class):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path="bert-base-chinese",
                                            cache_dir="./cache/bert-base-chinese/")
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_size, num_class)
    
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)
        x = out.last_hidden_state
        x = x[:, 0, :]
        x = self.fc(x)
        return x
    
model = Model(hidden_size=768, num_class=14)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

eval_every = 1

def eval(model, dataloader, device):
    model.eval()
    acc = 0
    tot = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, label = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            out = model(input_ids, attention_mask)
            predicted = torch.argmax(out, dim=-1)
            predicted = predicted.cpu()
            acc += (predicted == label).sum()
            tot += predicted.shape[0]
        return acc / tot
    
Epoch = 3
best_acc = 0.0
best_epoch = -1
early_stop = 5
for epoch in range(Epoch):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(train_dataloder):
        input_ids, attention_mask, label = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        predicted = model(input_ids, attention_mask)
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
        logger.info(f"epoch:{epoch}, valid acc: {acc * 100:.4f}%, best valid acc: {best_acc * 100:.4f}, at epoch {best_epoch}")

test_model = Model(hidden_size=768, num_class=14)
test_model.load_state_dict(torch.load(f"./pt/{model_name}.pt"))

test_model = test_model.to(device)

test_model.eval()
with torch.no_grad():
    acc = eval(test_model, test_dataloader, device)
    logger.info(f"test acc: {acc * 100:.4f}%")




