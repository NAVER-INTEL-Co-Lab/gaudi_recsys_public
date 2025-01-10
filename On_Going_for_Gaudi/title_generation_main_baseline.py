import os
import sys
import argparse

from utils import *
from title_generation_train_model_baseline import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # GPU train options
    parser.add_argument("--multi_gpu", action='store_true')
    # parser.add_argument('--device', type=str, default='hpu')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument("--world_size", type=int, default=8)
    # model setting
    parser.add_argument("--llm", type=str, default='llama-3b', help='flan_t5, llama, vicuna')
    parser.add_argument("--recsys", type=str, default='sasrec')
    parser.add_argument("--baseline", type=str, default='tallrec')

    # dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='Industrial_and_Scientific')
    # train phase setting
    parser.add_argument("--pretrain_stage2", action='store_true')
    # parser.add_argument("--pretrain_stage2", default=True)
    # parser.add_argument("--inference", default=True)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--load_candi", action='store_true')
    parser.add_argument("--save_dir", type=str, default='tallrecbest')


    # hyperparameters options
    parser.add_argument('--batch_size2', default=4, type=int)
    parser.add_argument('--batch_size_infer', default=16, type=int)
        
    parser.add_argument('--time', default=1, type=int)
    parser.add_argument('--infer_epoch', default=1, type=int)
    parser.add_argument('--maxlen', default=128, type=int)#50
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    parser.add_argument('--nn_parameter', default=False, action='store_true')

    parser.add_argument("--infer_type", type=str, default='original', help='original, transition, non_transition')
    parser.add_argument('--cold', default=0, type=int, help='1: warm 2:cold, 0:non')
    

    args = parser.parse_args()
    
    if args.device =='hpu':
        args.device = torch.device('hpu')
        args.is_hpu = True
        args.nn_parameter = True
    else:
        args.device = 'cuda:' + str(args.device)
        args.is_hpu = False
    
    if args.pretrain_stage2:
        train_model_phase2(args)
    elif args.inference:
        inference(args)