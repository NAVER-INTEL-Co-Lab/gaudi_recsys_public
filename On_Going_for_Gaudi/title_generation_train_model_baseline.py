import os
import torch
import random
import time
import os
import sys

from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR

from title_generation_models.title_generation_llmrec_model_baseline import *
from SeqRec.sasrec.utils import data_partition, SeqDataset, SeqDataset_Inference, SeqDataset_Validation
# from SeqRec.sasrec.utils_new import data_partition, SeqDataset, SeqDataset_Inference




def setup_ddp(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)
    if args.device.type == 'hpu':
        import habana_frameworks.torch.distributed.hccl
        init_process_group(backend="hccl", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    # htcore.set_device(rank)
        
def train_model_phase2(args):
    print(f'{args.baseline} strat train phase-2\n')
    if args.multi_gpu:
        world_size = args.world_size
        mp.spawn(train_model_phase2_,
             args=(world_size,args),
             nprocs=world_size,
             join=True)
    else:
        train_model_phase2_(0, 0, args)

def inference(args):
    print(f'{args.baseline} inference\n')
    if args.multi_gpu:
        world_size = args.world_size
        mp.spawn(inference_,
             args=(world_size,args),
             nprocs=world_size,
             join=True)
    else:
        inference_(0,0,args)
  

def train_model_phase2_(rank,world_size,args):
    if args.multi_gpu:
        setup_ddp(rank, world_size, args)
        if args.device == 'hpu':
            args.device = torch.device('hpu')
        else:
            args.device = 'cuda:' + str(rank)
    random.seed(0)

    model = llmrec_model(args).to(args.device)
    if args.baseline =='a-llmrec':
        model.load_model(args, phase1_epoch=5)
        
    dataset = data_partition(args.rec_pre_trained_data, args, path=f'./SeqRec/data_{args.rec_pre_trained_data}/{args.rec_pre_trained_data}')
    [user_train, user_valid, user_test, usernum, itemnum, eval_set, eval_item_set] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size2
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    train_data_set = SeqDataset(user_train, len(user_train.keys()), itemnum, args.maxlen)
    
    valid_data_set = SeqDataset_Validation(user_train, user_valid, list(user_valid.keys()), itemnum, args.maxlen)
    
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size2, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        valid_data_loader = DataLoader(valid_data_set, batch_size = args.batch_size_infer, sampler=DistributedSampler(valid_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size2, pin_memory=True, shuffle=True)
        valid_data_loader = DataLoader(valid_data_set, batch_size = args.batch_size_infer, pin_memory=True, shuffle=True)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.98))
    scheduler = LambdaLR(adam_optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    epoch_start_idx = 1
    T = 0.0
    best_perform = 0
    early_stop = 0
    t0 = time.time()
    
    eval_set_use = eval_set[1]
    if len(eval_set_use)>10000:
        users = random.sample(list(eval_set_use), 10000)
    else:
        users = list(eval_set_use)
    
    user_list = []
    for u in users:
        if len(user_test[u]) < 1: continue
        user_list.append(u)

    inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, args.maxlen)
    if args.multi_gpu:
        inference_data_loader = DataLoader(inference_data_set, batch_size = args.batch_size_infer, sampler=DistributedSampler(inference_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, static_graph=True)
    else:
        inference_data_loader = DataLoader(inference_data_set, batch_size = args.batch_size_infer, pin_memory=True)
    model.args.llara_thres = 0
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        model.train()
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)
        for step, data in enumerate(train_data_loader):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            if epoch >1:
                model.args.llara_thres = 1
            else:
                if args.baseline == 'llara':
                    model.args.llara_thres = step / num_batch
                    
            model([u,seq,pos,neg], optimizer=adam_optimizer, batch_iter=[epoch,args.num_epochs + 1,step,num_batch], mode='phase2')
            
            if step % (num_batch//5) ==0 and step !=0:
                if rank ==0:
                    if args.multi_gpu: 
                        model.module.save_model(args, epoch2=epoch, best=True)
                    else:
                        if args.baseline =='a-llmrec':
                            model.save_model(args,  epoch1= 5, epoch2=epoch)
                        else:
                            model.save_model(args,  epoch2=epoch, best=True)

        #     if step % (num_batch//5) ==0 and step !=0:
                
        #         model.eval()
        #         num_valid_batch = len(user_valid.keys())//args.batch_size_infer
        #         model.users = 0.0
        #         model.NDCG = 0.0
        #         model.HT = 0.0
        #         model.NDCG_20 = 0.0
        #         model.HIT_20 = 0.0
        #         model.all_embs = None
        #         with torch.no_grad():
        #             for _, data in enumerate(valid_data_loader):
        #                 # if _ > int(num_valid_batch*0.1):
        #                 #     break
        #                 u, seq, pos, neg = data
        #                 u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()                        
        #                 # model([u,seq,pos,neg, rank, False, None, 'original'], mode='generate')
        #                 model([u,seq,pos,neg, rank, False, None, 'original'], mode='generate_batch')
                        
        #         perform = model.HT/model.users
        #         if perform >= best_perform:
        #             best_perform = perform
        #             if rank ==0:
        #                 if args.multi_gpu: model.module.save_model(args, epoch2=epoch, best=True)
        #                 else: model.save_model(args,  epoch2=epoch, best=True)
                    
        #             model.users = 0.0
        #             model.NDCG = 0.0
        #             model.HT = 0.0
        #             model.NDCG_20 = 0.0
        #             model.HIT_20 = 0.0
        #             with torch.no_grad():
        #                 for _, data in enumerate(inference_data_loader):
        #                     # if _ > int(num_valid_batch*0.1):
        #                     #     break
        #                     u, seq, pos, neg = data
        #                     u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()                        
        #                     # model([u,seq,pos,neg, rank, False, None, 'original'], mode='generate')
        #                     model([u,seq,pos,neg, rank, False, None, 'original'], mode='generate_batch')
        #             out_dir = f'./models/{args.save_dir}/'
        #             out_dir = out_dir[:-1] + 'best/'
                    
        #             out_dir += f'{args.rec_pre_trained_data}_'
                    
        #             out_dir += f'{args.llm}_{epoch}_results.txt'
                    
        #             f = open(out_dir, 'a')
        #             f.write(f'NDCG: {model.NDCG/model.users}, HR: {model.HT/model.users}\n')
        #             f.write(f'NDCG20: {model.NDCG_20/model.users}, HR20: {model.HIT_20/model.users}\n')
        #             f.close()
                    
        #             early_stop = 0
        #         else:
        #             model.save_model(args,  epoch2=epoch)
        #             early_stop +=1
        #         if early_stop == 5:
        #             sys.exit("Terminating Train")
        #         model.train()
        #         scheduler.step()
                
        # if rank == 0:
        #     model.eval()
        #     num_valid_batch = len(user_valid.keys())//args.batch_size_infer
        #     model.users = 0.0
        #     model.NDCG = 0.0
        #     model.HT = 0.0
        #     model.NDCG_20 = 0.0
        #     model.HIT_20 = 0.0
        #     model.all_embs = None
        #     with torch.no_grad():
        #         for _, data in enumerate(valid_data_loader):
        #             # if _ > int(num_valid_batch*0.1):
        #             #     break
        #             u, seq, pos, neg = data
        #             u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()                        
        #             # model([u,seq,pos,neg, rank, False, None, 'original'], mode='generate')
        #             model([u,seq,pos,neg, rank, False, None, 'original'], mode='generate_batch')
                    
        #     perform = model.HT/model.users
        #     if perform >= best_perform:
        #         best_perform = perform
        #         if rank ==0:
        #             if args.multi_gpu: model.module.save_model(args, epoch2=epoch, best=True)
        #             else: model.save_model(args,  epoch2=epoch, best=True)
                
        #         model.users = 0.0
        #         model.NDCG = 0.0
        #         model.HT = 0.0
        #         model.NDCG_20 = 0.0
        #         model.HIT_20 = 0.0
        #         with torch.no_grad():
        #             for _, data in enumerate(inference_data_loader):
        #                 # if _ > int(num_valid_batch*0.1):
        #                 #     break
        #                 u, seq, pos, neg = data
        #                 u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()                        
        #                 # model([u,seq,pos,neg, rank, False, None, 'original'], mode='generate')
        #                 model([u,seq,pos,neg, rank, False, None, 'original'], mode='generate_batch')
        #         out_dir = f'./models/{args.save_dir}/'
        #         out_dir = out_dir[:-1] + 'best/'
                
        #         out_dir += f'{args.rec_pre_trained_data}_'
                
        #         out_dir += f'{args.llm}_{epoch}_results.txt'
                
        #         f = open(out_dir, 'a')
        #         f.write(f'NDCG: {model.NDCG/model.users}, HR: {model.HT/model.users}\n')
        #         f.write(f'NDCG20: {model.NDCG_20/model.users}, HR20: {model.HIT_20/model.users}\n')
        #         f.close()

                
        #         early_stop = 0
        #     else:
        #         model.save_model(args,  epoch2=epoch)
        #         early_stop +=1
        #     if early_stop == 2:
        #         sys.exit("Terminating Train")
        #     model.train()
        #     scheduler.step()

    
    print('phase2 train time :', time.time() - t0)
    if args.multi_gpu:
        destroy_process_group()
    return

def inference_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = torch.device('hpu')
        
    model = llmrec_model(args).to(args.device)
    phase2_epoch = args.infer_epoch
    if args.baseline =='a-llmrec':
        model.load_model(args, phase1_epoch=5, phase2_epoch=phase2_epoch)
    else:
        model.load_model(args, phase2_epoch=phase2_epoch)

    dataset = data_partition(args.rec_pre_trained_data, args, path=f'./SeqRec/data_{args.rec_pre_trained_data}/{args.rec_pre_trained_data}')
    [user_train, user_valid, user_test, usernum, itemnum, eval_set, eval_item_set] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size_infer
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    model.eval()
    
    print("Eval Original Set")
    eval_set_use = eval_set[1]
    shuffle = False
    candi_set = None
    files = 'original'
    
    if len(eval_set_use)>10000:
        users = random.sample(list(eval_set_use), 10000)
    else:
        users = list(eval_set_use)
        
    if args.cold > 0:
        f = open('./SeqRec/data_Video_Games/Video_Games_train.txt', 'r')#processed 로 하면 성능 개 높음? 뭐임?
        lines = f.readlines()
        from collections import defaultdict
        item_count = defaultdict(int)
        for line in lines:
            u, i = line.strip().split(' ')
            item_count[int(i)] +=1
        f.close()
        users = []
        all_int = sum(item_count.values())
        if args.cold == 1:
            coldwarm = True
        else:
            coldwarm = False
        l = sorted(item_count.items(), key=lambda item: item[1], reverse=coldwarm)
        rate = 0.0
        items = []
        for l_ in l:
            if rate >0.5:
                break
            rate += l_[1]/all_int
            items.append(l_[0])
        for u in eval_set_use:
            if user_test[u][0] in items:
                users.append(u)
        random.shuffle(users)
        users = users[:10000]
        print('users: ', len(users))
        candi_set = None
    
    # user_list = []
    # for u in users:
    #     if len(user_train[u]) < 1 or len(user_test[u]) < 1: continue
    #     user_list.append(u)
        
    user_list = []
    for u in users:
        if len(user_test[u]) < 1: continue
        user_list.append(u)

    if args.load_candi:
        user_list = list(model.candi_set.keys())
    
    # inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, args.maxlen)
    inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, list(user_test.keys()), itemnum, args.maxlen)

    if args.multi_gpu:
        inference_data_loader = DataLoader(inference_data_set, batch_size = args.batch_size_infer, sampler=DistributedSampler(inference_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, static_graph=True)
    else:
        inference_data_loader = DataLoader(inference_data_set, batch_size = args.batch_size_infer, pin_memory=True)

    if not os.path.exists(f'./{files}_results'):
        os.makedirs(f'./{files}_results')

    for _, data in enumerate(inference_data_loader):
        u, seq, pos, neg = data
        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
        model([u,seq,pos,neg, rank, shuffle, candi_set, files], mode='generate')
        
        # model([u,seq,pos,neg, rank, shuffle, candi_set, files], mode='generate_batch')
        # model([u,seq,pos,neg, rank, shuffle, candi_set, files], mode='generate_all')