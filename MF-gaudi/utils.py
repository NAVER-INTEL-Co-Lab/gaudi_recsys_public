import random
import numpy as np
from torch.utils.data import Dataset

def data_partition(fname, path=None):
    usernum = 0
    itemnum = 0
    datas = []
    # f = open('./pre_train/sasrec/data/%s.txt' % fname, 'r')
    if path == None:
        f = open('./../SASRec-gaudi/data/processed/%s.txt' % fname, 'r')
    else:
        f = open(path, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)

        datas.append([u-1,i-1])
    
    random.shuffle(datas)
    
    label_list = np.zeros((usernum,itemnum))
    for d in datas:
        label_list[d[0],d[1]] = 1
    
    train = int(len(datas)*0.8)
    val = int(len(datas)*0.1)
    datas = np.array(datas)
    user_train = datas[:train]
    user_valid = datas[train:train+val]
    user_test = datas[train+val:]
    
    return user_train, user_valid, user_test, usernum, itemnum, label_list

class SeqDataset(Dataset):
    def __init__(self, datas): 
        self.datas = datas
        self.len_data = len(datas)

    def __len__(self):
        return self.len_data
        
    def __getitem__(self, idx):
        
        data = self.datas[idx]
        user_id = data[0]
        item_id = data[1]

        return user_id, item_id
    
def evaluate(model, dataset, args):
    
    acc = 0
    total_len = 0
    for step, data in enumerate(dataset):
        
        results = model.predict(data[0],data[1])
        acc +=results[0]
        total_len +=results[1]
    
    return acc/total_len