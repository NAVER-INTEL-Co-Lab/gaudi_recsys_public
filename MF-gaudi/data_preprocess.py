import os
import os.path
import gzip
import json
from tqdm import tqdm
from collections import defaultdict

from datasets import load_dataset

        
def preprocess(fname):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{fname}", trust_remote_code=True)

    dataset = dataset['full']
    # counting interactions for each user and item
    
    for l in tqdm(dataset):
        line += 1
        asin = l['asin']
        rev = l['user_id']
        time = l['timestamp']
        countU[rev] += 1
        countP[asin] += 1
    
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    
    
    for l in tqdm(dataset):
        line += 1
        asin = l['asin']
        rev = l['user_id']
        time = l['timestamp']
        
        threshold = 5
            
        if countU[rev] < threshold or countP[asin] < threshold:
            continue
        
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
        
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])
    
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])
        
    print(usernum, itemnum)
    
    f = open(f'./../SASRec-gaudi/data/processed/{fname}.txt', 'w')
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1]))
    f.close()
    
    del dataset