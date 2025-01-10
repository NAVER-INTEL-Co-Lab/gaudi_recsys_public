import os
import time
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from data_preprocess import *
from utils import *

###GAUDI
import habana_frameworks.torch.core as htcore

from models.pmf import PMF

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--l2', default=0.0, type=float)
    parser.add_argument('--device', default='hpu', type=str, help='cpu, hpu')
    parser.add_argument('--inference_only', default=False, action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--nn_parameter', default=False, action='store_true')
    
    args = parser.parse_args()
    
    if not os.path.isfile(f'./../SASRec-gaudi/data/processed/{args.dataset}.txt'):
        print("Download Dataset")
        preprocess(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum, label_matrix = data_partition(args.dataset)
    
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size
    
    if args.device =='hpu':
        args.device = torch.device('hpu')

    train_dataset = SeqDataset(user_train)
    valid_dataset = SeqDataset(user_valid)
    test_dataset = SeqDataset(user_test)
    
    train_data_loader = DataLoader(train_dataset, batch_size = args.batch_size, pin_memory=True, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size = args.batch_size, pin_memory=True, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size = args.batch_size, pin_memory=True, shuffle=True)

    label_matrix = torch.LongTensor(label_matrix).to(args.device)
    
    model = PMF(usernum, itemnum, label_matrix, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
        
    epoch_start_idx = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    time_list = []
    loss_list = []
    T = 0.0
    t0 = time.time()
    start_time = time.time()
    
    
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        model.train()
        epoch_s_time = time.time()
        total_loss, count = 0, 0
        for step, data in enumerate(train_data_loader):
            
            loss = model(data[0],data[1])
            
            #GAUDI
            loss.backward()
            if args.device =='hpu':
                htcore.mark_step()
            optimizer.step()
            if args.device =='hpu':
                htcore.mark_step()
                
            total_loss += loss.item()
            count+=1

            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
        epoch_e_time = time.time()
        time_list.append(epoch_e_time - epoch_s_time)
        loss_list.append(total_loss/count)

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            with torch.no_grad():
                t_test = evaluate(model, test_data_loader, args)
                t_valid = evaluate(model, valid_data_loader, args)
            print('\n')
            print('epoch:%d, time: %f(s), valid (ACC: %.4f), test (ACC: %.4f)'
                    % (epoch, T, t_valid, t_test))

            print(str(t_valid) + ' ' + str(t_test) + '\n')
            t0 = time.time()
            model.train()
    end_time = time.time()
    print("Done")
    print("Time:", end_time-start_time)
    
    np.save('time.npy', np.array(time_list))
    np.save('loss.npy', np.array(loss_list))