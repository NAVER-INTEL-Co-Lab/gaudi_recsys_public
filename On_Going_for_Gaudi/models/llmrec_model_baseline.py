import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.llm4rec_baseline import *
from sentence_transformers import SentenceTransformer
from datetime import datetime

from tqdm import trange, tqdm

try:
    import habana_frameworks.torch.core as htcore
except:
    0
    
class two_layer_mlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1


class llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        
        with open(f'./SeqRec/data_{args.rec_pre_trained_data}/text_name_dict.json.gz','rb') as ft:
            self.text_name_dict = pickle.load(ft)
            
        if args.load_candi:
            with open(f'./SeqRec/data_{args.rec_pre_trained_data}/candi_set.json.gz','rb') as ft:
                self.candi_set = pickle.load(ft)
        
        if args.testing_sasrec:
            paths = rec_pre_trained_data + '_trainshuffle'
            print(paths)
            self.recsys = RecSys(args.recsys, paths, self.device)
        else:
            self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768
        
        if self.args.nn_parameter == True:
            self.recsys_mean = self.recsys.model.item_emb.data.mean(axis=0)
            self.recsys_std = self.recsys.model.item_emb.data.std(axis=0)
        else:
            self.recsys_mean = self.recsys.model.item_emb.weight.mean(axis=0)
            self.recsys_std = self.recsys.model.item_emb.weight.std(axis=0)
                
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.all_embs = None
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG=0
        self.lan_HIT=0
        self.NDCG_20 = 0
        self.HIT_20 = 0
        self.num_user = 0
        self.yes = 0

        self.extract_embs_list = []
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        if args.baseline =='a-llmrec':
            self.mlp = two_layer_mlp(self.rec_sys_dim)
            if args.pretrain_stage1:
                self.sbert = SentenceTransformer('nq-distilbert-base-v1')
                self.mlp2 = two_layer_mlp(self.sbert_dim)
        
        
        if args.pretrain_stage2 or args.inference or args.extract:
            
            if args.inference or args.extract:
                epochs = args.infer_epoch
                out_dir = f'./models/{args.save_dir}/{args.rec_pre_trained_data}_' + f'{args.llm}_{epochs}_'
                if self.args.baseline != 'a-llmrec':
                    self.llm = llm4rec(device=self.device, llm_model=args.llm, load_lora=True, load_config=out_dir, args = self.args)
                else:
                    self.llm = llm4rec(device=self.device, llm_model=args.llm, args = self.args)
            else:
                self.llm = llm4rec(device=self.device, llm_model=args.llm, args = self.args)
            
            
            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                # nn.BatchNorm1d(self.llm.llm_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            )
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            if self.args.baseline == 'a-llmrec':
                self.item_emb_proj = nn.Sequential(
                    nn.Linear(128, self.llm.llm_model.config.hidden_size),
                    nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                    # nn.BatchNorm1d(self.llm.llm_model.config.hidden_size),
                    # nn.GELU(),
                    nn.LeakyReLU(),
                    nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
                )
                nn.init.xavier_normal_(self.item_emb_proj[0].weight)
                nn.init.xavier_normal_(self.item_emb_proj[3].weight)
            else:
                self.item_emb_proj = nn.Sequential(
                    nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
                    nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                    # nn.BatchNorm1d(self.llm.llm_model.config.hidden_size),
                    # nn.GELU(),
                    nn.LeakyReLU(),
                    nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
                )
                nn.init.xavier_normal_(self.item_emb_proj[0].weight)
                nn.init.xavier_normal_(self.item_emb_proj[3].weight)
                
            self.users = 0.0
            self.NDCG = 0.0
            self.HT = 0.0
            
    def standard(self, emb):
        
        return (emb-self.recsys_mean)/(self.recsys_std+1e-8)
            
    def save_model(self, args, epoch1 = None, epoch2=None, best=False):
        # out_dir = f'./models/saved_models/'
        # out_dir = f'./models/saved_models_new_loss_three_objective2_infonce/'
        # out_dir = f'./models/saved_models_new_loss_three_infonce_mse/'
        # out_dir = f'./models/saved_models_original_raw_video_re/'
        out_dir = f'./models/{args.save_dir}/'
        if best:
            out_dir = out_dir[:-1] + 'best/'
        
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_'
        if args.baseline =='a-llmrec':
            out_dir += f'{epoch1}_'
            if args.pretrain_stage1:
                torch.save(self.sbert.state_dict(), out_dir + 'sbert.pt')
                torch.save(self.mlp.state_dict(), out_dir + 'mlp.pt')
                torch.save(self.mlp2.state_dict(), out_dir + 'mlp2.pt')
            elif args.pretrain_stage2:
                out_dir += f'{args.llm}_{epoch2}_'
                torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
                torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
                
                torch.save(self.llm.pred_user.state_dict(), out_dir + 'pred_user.pt')
                torch.save(self.llm.pred_item.state_dict(), out_dir + 'pred_item.pt')
                
                if args.nn_parameter:
                    torch.save(self.llm.CLS, out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item, out_dir + 'CLS_item.pt')
                else:
                    torch.save(self.llm.CLS.state_dict(), out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.state_dict(), out_dir + 'CLS_item.pt')
        else:
            out_dir += f'{args.llm}_{epoch2}_'
            if args.pretrain_stage2:
                if args.baseline != 'tallrec':
                    torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
                    torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
                
                torch.save(self.llm.pred_user.state_dict(), out_dir + 'pred_user.pt')
                torch.save(self.llm.pred_item.state_dict(), out_dir + 'pred_item.pt')
                
                if args.nn_parameter:
                    torch.save(self.llm.CLS, out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item, out_dir + 'CLS_item.pt')
                else:
                    torch.save(self.llm.CLS.state_dict(), out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.state_dict(), out_dir + 'CLS_item.pt')
                if self.args.is_hpu:
                    self.llm.llm_model.to('cpu')
                    self.llm.llm_model.save_pretrained(out_dir, save_embedding_layers=True)
                    self.llm.llm_model.to(args.device)
                else:
                    self.llm.llm_model.save_pretrained(out_dir, save_embedding_layers=True)

            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        if args.baseline =='a-llmrec' and args.pretrain_stage2:
            out_dir = f'./models/{args.save_dir}best/{args.rec_pre_trained_data}_'
        else:
            out_dir = f'./models/{args.save_dir}/{args.rec_pre_trained_data}_'
        if args.baseline =='a-llmrec':
            out_dir += f'{phase1_epoch}_'
            mlp = torch.load(out_dir + 'mlp.pt', map_location = args.device)
            self.mlp.load_state_dict(mlp)
            del mlp
            for name, param in self.mlp.named_parameters():
                param.requires_grad = False
                
        if args.inference or args.extract:
            out_dir += f'{args.llm}_{phase2_epoch}_'
            
            if args.baseline != 'tallrec':
                if args.is_hpu:
                    log_emb_proj = torch.load(out_dir + 'log_proj.pt')
                    self.log_emb_proj.load_state_dict(log_emb_proj)
                    del log_emb_proj
                    
                    item_emb_proj = torch.load(out_dir + 'item_proj.pt')
                    self.item_emb_proj.load_state_dict(item_emb_proj)
                    del item_emb_proj
                else:
                    log_emb_proj = torch.load(out_dir + 'log_proj.pt', map_location = self.device)
                    self.log_emb_proj.load_state_dict(log_emb_proj)
                    del log_emb_proj
                    
                    item_emb_proj = torch.load(out_dir + 'item_proj.pt', map_location = self.device)
                    self.item_emb_proj.load_state_dict(item_emb_proj)
                    del item_emb_proj
            if args.is_hpu:
                pred_user = torch.load(out_dir + 'pred_user.pt')
                self.llm.pred_user.load_state_dict(pred_user)
                del pred_user
                
                pred_item = torch.load(out_dir + 'pred_item.pt')
                self.llm.pred_item.load_state_dict(pred_item)
                del pred_item

                if args.nn_parameter:
                    CLS = torch.load(out_dir + 'CLS.pt')
                    self.llm.CLS.data = CLS.data
                    del CLS
                    
                    CLS_item = torch.load(out_dir + 'CLS_item.pt')
                    self.llm.CLS_item.data = CLS_item.data
                    del CLS_item
                else:
                    CLS = torch.load(out_dir + 'CLS.pt')
                    self.llm.CLS.load_state_dict(CLS)
                    del CLS
                    
                    CLS_item = torch.load(out_dir + 'CLS_item.pt')
                    self.llm.CLS_item.load_state_dict(CLS_item)
                    del CLS_item
            else:
                pred_user = torch.load(out_dir + 'pred_user.pt', map_location = self.device)
                self.llm.pred_user.load_state_dict(pred_user)
                del pred_user
                
                pred_item = torch.load(out_dir + 'pred_item.pt', map_location = self.device)
                self.llm.pred_item.load_state_dict(pred_item)
                del pred_item

                CLS = torch.load(out_dir + 'CLS.pt', map_location = self.device)
                self.llm.CLS.load_state_dict(CLS)
                del CLS
                
                CLS_item = torch.load(out_dir + 'CLS_item.pt', map_location = self.device)
                self.llm.CLS_item.load_state_dict(CLS_item)
                del CLS_item
            
            # token = torch.load(out_dir + 'token.pt', map_location = self.device)
            
            # if self.args.baseline != 'a-llmrec':
            #     self.llm.llm_model.base_model.model.model.embed_tokens.load_state_dict(token)
            # else:
            #     self.llm.llm_model.model.embed_tokens.load_state_dict(token)
            # del token
            

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]
        
    def find_item_time(self, item, user, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        l = [datetime.utcfromtimestamp(int(self.text_name_dict['time'][i][int(user)])/1000) for i in item]
        return [l_.strftime('%Y-%m-%d') for l_ in l]
    

    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'
        
    def get_item_emb(self, item_ids):
        with torch.no_grad():
            if self.args.nn_parameter:
                item_embs = self.recsys.model.item_emb[torch.LongTensor(item_ids).to(self.device)]
            else:
                item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
                
            if self.args.baseline =='a-llmrec':
                item_embs, _ = self.mlp(item_embs)
        
        return item_embs
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase1':
            self.pre_train_phase1(data, optimizer, batch_iter)
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode =='generate':
            self.generate(data)
            print('test (NDCG@10: %.4f, HR@10: %.4f), Num User: %.4f'
                    % (self.NDCG/self.users, self.HT/self.users, self.users))
        if mode =='generate_all':
            self.generate_all(data)
            print('test (NDCG@10: %.4f, HR@10: %.4f), Num User: %.4f'
                    % (self.NDCG/self.users, self.HT/self.users, self.users))
        if mode=='generate_batch':
            self.generate_batch(data)
            print(self.args.save_dir, self.args.rec_pre_trained_data)
            print('test (NDCG@10: %.4f, HR@10: %.4f), Num User: %.4f'
                    % (self.NDCG/self.users, self.HT/self.users, self.users))
            print('test (NDCG@20: %.4f, HR@20: %.4f), Num User: %.4f'
                    % (self.NDCG_20/self.users, self.HIT_20/self.users, self.users))
        if mode=='extract':
            self.extract_emb(data)

    def make_interact_text(self, interact_ids, interact_max_num, user, shuffle = False, llara_text = False, original_seq = None, time = True):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        
        # if shuffle == True:
        #     if interact_max_num =='all':
        #         times = self.find_item_time(original_seq, user)
        #     else:
        #         times = self.find_item_time(original_seq[-interact_max_num:], user)
        # else:
        #     if interact_max_num =='all':
        #         times = self.find_item_time(interact_ids, user)
        #     else:
        #         times = self.find_item_time(interact_ids[-interact_max_num:], user)
                
        if shuffle == True:
            times = self.find_item_time(original_seq, user)
        else:
            times = self.find_item_time(interact_ids, user)
            
        interact_text = []
        count = 1
        if interact_max_num == 'all':
            for title in interact_item_titles_:
                if self.args.baseline == 'tallrec' or (self.args.baseline =='llara' and llara_text):
                    if time:
                        interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title)
                    else:
                        interact_text.append(title)
                else:
                    if time:
                        interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                    else:
                        interact_text.append(title + '[HistoryEmb]')
                count+=1
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                if self.args.baseline == 'tallrec' or (self.args.baseline =='llara' and llara_text):
                    if time:
                        interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title)
                    else:
                        interact_text.append(title)
                else:
                    if time:
                        interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                    else:
                        interact_text.append(title + '[HistoryEmb]')
                count+=1
            interact_ids = interact_ids[-interact_max_num:]
                    
            
        interact_text = ','.join(interact_text)
        return interact_text, interact_ids
    
    
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set = None, llara_text = False):
        neg_item_id = []
        if candi_set == None:
            neg_item_id = []
            while len(neg_item_id)<99:
                t = np.random.randint(1, self.item_num+1)
                if not (t in interact_ids or t in neg_item_id):
                    neg_item_id.append(t)
        else:
            his = set(interact_ids)
            items = list(candi_set.difference(his))
            if len(items) >99:
                neg_item_id = random.sample(items, 99)
            else:
                while len(neg_item_id)<49:
                    t = np.random.randint(1, self.item_num+1)
                    if not (t in interact_ids or t in neg_item_id):
                        neg_item_id.append(t)
        random.shuffle(neg_item_id)
        
        candidate_ids = [target_item_id]
        # candidate_text = [target_item_title + '[CandidateEmb]']HistoryEmb
        
        if self.args.baseline == 'tallrec' or (self.args.baseline =='llara' and llara_text):
            candidate_text = [f'The item title is as follows: ' + target_item_title + ", then generate item representation token:[ItemOut][ItemOut][ItemOut]"]
        else:
            candidate_text = [f'The item title and item embedding are as follows: ' + target_item_title + "[HistoryEmb], then generate item representation token:[ItemOut][ItemOut][ItemOut]"]


        for neg_candidate in neg_item_id[:candidate_num - 1]:
            if self.args.baseline == 'tallrec' or (self.args.baseline =='llara' and llara_text):
                candidate_text.append(f'The item title is as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + ", then generate item representation token:[ItemOut][ItemOut][ItemOut]")
            else:
                candidate_text.append(f'The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut][ItemOut][ItemOut]")

            candidate_ids.append(neg_candidate)
                
        # random_ = np.random.permutation(len(candidate_text))
        # candidate_text = np.array(candidate_text)[random_]
        # candidate_ids = np.array(candidate_ids)[random_]
            
        return candidate_text, candidate_ids
    
    
    def make_candidate(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set = None, llara_text = False):
        neg_item_id = []
        if candi_set == None:
            neg_item_id = []
            while len(neg_item_id)<99:
                t = np.random.randint(1, self.item_num+1)
                if not (t in interact_ids or t in neg_item_id):
                    neg_item_id.append(t)
        else:
            his = set(interact_ids)
            items = list(candi_set.difference(his))
            if len(items) >99:
                neg_item_id = random.sample(items, 99)
            else:
                while len(neg_item_id)<99:
                    t = np.random.randint(1, self.item_num+1)
                    if not (t in interact_ids or t in neg_item_id):
                        neg_item_id.append(t)
        random.shuffle(neg_item_id)
        
        candidate_ids = [target_item_id]
        
        candidate_ids = candidate_ids + neg_item_id[:candidate_num - 1]
            
        return candidate_ids
    
    
    
    
    def pre_train_phase1(self,data,optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        print('Pretrain stage1', self.args.save_dir, self.args.rec_pre_trained_data, self.args.llm)
        self.sbert.train()
        optimizer.zero_grad()

        u, seq, pos, neg = data
        indices = [self.maxlen*(i+1)-1 for i in range(u.shape[0])]
    
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')
        log_emb_ = log_emb[indices]
        pos_emb_ = pos_emb[indices]
        neg_emb_ = neg_emb[indices]
        pos_ = pos.reshape(pos.size)[indices]
        neg_ = neg.reshape(neg.size)[indices]
        
        start_inx = 0
        end_inx = 60
        iterss = 0
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0
        rc_loss = 0
        text_rc_loss = 0
        original_loss = 0
        while start_inx < len(log_emb_):
            log_emb = log_emb_[start_inx:end_inx]
            pos_emb = pos_emb_[start_inx:end_inx]
            neg_emb = neg_emb_[start_inx:end_inx]
            
            pos__ = pos_[start_inx:end_inx]
            neg__ = neg_[start_inx:end_inx]
            
            start_inx = end_inx
            end_inx += 60
            iterss +=1
            
            pos_text = self.find_item_text(pos__)
            neg_text = self.find_item_text(neg__)
            
            pos_token = self.sbert.tokenize(pos_text)
            pos_text_embedding= self.sbert({'input_ids':pos_token['input_ids'].to(self.device),'attention_mask':pos_token['attention_mask'].to(self.device)})['sentence_embedding']
            neg_token = self.sbert.tokenize(neg_text)
            neg_text_embedding= self.sbert({'input_ids':neg_token['input_ids'].to(self.device),'attention_mask':neg_token['attention_mask'].to(self.device)})['sentence_embedding']
            
            pos_text_matching, pos_proj = self.mlp(pos_emb)
            neg_text_matching, neg_proj = self.mlp(neg_emb)
            
            pos_text_matching_text, pos_text_proj = self.mlp2(pos_text_embedding)
            neg_text_matching_text, neg_text_proj = self.mlp2(neg_text_embedding)
            
            pos_logits, neg_logits = (log_emb*pos_proj).mean(axis=1), (log_emb*neg_proj).mean(axis=1)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=pos_logits.device), torch.zeros(neg_logits.shape, device=pos_logits.device)

            loss = self.bce_criterion(pos_logits, pos_labels)
            loss += self.bce_criterion(neg_logits, neg_labels)
            
            matching_loss = self.mse(pos_text_matching,pos_text_matching_text) + self.mse(neg_text_matching,neg_text_matching_text)
            reconstruction_loss = self.mse(pos_proj,pos_emb) + self.mse(neg_proj,neg_emb)
            text_reconstruction_loss = self.mse(pos_text_proj,pos_text_embedding.data) + self.mse(neg_text_proj,neg_text_embedding.data)
            
            total_loss = loss + matching_loss + 0.5*reconstruction_loss + 0.2*text_reconstruction_loss
            total_loss.backward()
            optimizer.step()
            
            mean_loss += total_loss.item()
            bpr_loss += loss.item()
            gt_loss += matching_loss.item()
            rc_loss += reconstruction_loss.item()
            text_rc_loss += text_reconstruction_loss.item()
            
        print("loss in epoch {}/{} iteration {}/{}: {} / BPR loss: {} / Matching loss: {} / Item reconstruction: {} / Text reconstruction: {}".format(epoch, total_epoch, step, total_step, mean_loss/iterss, bpr_loss/iterss, gt_loss/iterss, rc_loss/iterss, text_rc_loss/iterss))

    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        print(self.args.save_dir, self.args.rec_pre_trained_data, self.args.llm)
        optimizer.zero_grad()
        u, seq, pos, neg = data
        
        
        mean_loss = 0
        
        text_input = []
        candidates_pos = []
        candidates_neg = []
        interact_embs = []
        candidate_embs_pos = []
        candidate_embs_neg = []
        candidate_embs = []
        
        loss_rm_mode1 = 0
        # self.llm.eval()
        
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
        
        llara_ind = []
        llara_ind_item = []
        for i in range(len(u)):
            target_item_id = pos[i][-1]
            llara_text = False
            if self.args.baseline == 'llara':
                rand = random.random()
                if rand > self.args.llara_thres:
                    llara_text = True
                    llara_ind.append(1)
                else:
                    llara_ind.append(0)
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
            

            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], llara_text=llara_text,time = self.args.time)
                
            candidate_num = 4
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, llara_text=llara_text)

            if self.args.baseline == 'llara':
                for __ in range(candidate_num):
                    llara_ind_item.append(1)
            
            if self.args.baseline == 'tallrec' or (self.args.baseline =='llara'):
                input_text = ''
            else:
                input_text = 'This is user representation from recommendation models : [UserRep]. '

            input_text += 'This user has made a series of purchases in the following order: '
                
            input_text += interact_text
            
            input_text +=". Based on this user representation from recommendation model and sequence of purchases, to recommend one next item for this user, generate user representation token:[UserOut][UserOut][UserOut]"


            text_input.append(input_text)
            
            candidates_pos += candidate_text             
            
            
            interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
            candidate_embs_pos.append(self.item_emb_proj((self.get_item_emb([candidate_ids]))).squeeze(0))
                        
        
        candidate_embs = torch.cat(candidate_embs_pos)
        
        log_emb = self.log_emb_proj(log_emb)
        
        if self.args.baseline != 'llara':
            llara_ind = None
            llara_ind_item = None
        
        samples = {'text_input': text_input, 'log_emb':log_emb, 'candidates_pos': candidates_pos, 'interact': interact_embs, 'candidate_embs':candidate_embs, 'llara_ind': llara_ind, 'llara_ind_item':llara_ind_item}
        
        

        loss_rm_mode1 = self.llm(samples, mode=0)

        mean_loss += loss_rm_mode1.item()
                    
        print("{} model loss in epoch {}/{} iteration {}/{}: {}".format(self.args.baseline, epoch, total_epoch, step, total_step, mean_loss))
        
        
        loss = loss_rm_mode1
        loss.backward()
        if self.args.is_hpu:
            htcore.mark_step()
        optimizer.step()
        if self.args.is_hpu:
            htcore.mark_step()
        
    
    def split_into_batches(self,itemnum, m):
        numbers = list(range(1, itemnum+1))
        
        # 전체 데이터를 batch_size 크기로 나누기
        batches = [numbers[i:i + m] for i in range(0, itemnum, m)]
        
        return batches

    
    def generate(self,data):
        if self.args.is_hpu:
            dev = 'hpu'
        else:
            dev = 'cuda'
        if self.all_embs == None:
            batches = self.split_into_batches(self.item_num, 128)#128
            self.all_embs = []
            max_input_length = 1024
            for bat in tqdm(batches):
                candidate_text = []
                candidate_ids = []
                candidate_embs = []
                for neg_candidate in bat:
                    if self.args.baseline == 'tallrec':
                        candidate_text.append('The item title is as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + ", then generate item representation token:[ItemOut][ItemOut][ItemOut]")
                    else:
                        candidate_text.append('The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut][ItemOut][ItemOut]")
                    
                    candidate_ids.append(neg_candidate)
                with torch.no_grad():
                    candi_tokens = self.llm.llm_tokenizer(
                        candidate_text,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=max_input_length,
                    ).to(self.device)
                    candidate_embs.append(self.item_emb_proj((self.get_item_emb(candidate_ids))))

                    candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                    if self.args.baseline == 'tallrec':
                        candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemOut]'], embs= {})
                    else:
                        candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemOut]', '[HistoryEmb]'], embs= {'[HistoryEmb]':candidate_embs[0]})
                    with torch.amp.autocast(dev):
                        candi_outputs = self.llm.llm_model.forward(
                            inputs_embeds=candi_embeds,
                            # attention_mask=attention_mask,
                            # return_dict=True,
                            # labels=targets,
                            output_hidden_states=True
                        )
                        
                        indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]')
                        # item_outputs = torch.stack([candi_outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))]).squeeze(1)
                        item_outputs = torch.cat([candi_outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])

                        item_outputs = self.llm.pred_item(item_outputs)
                    
                    self.all_embs.append(item_outputs)
                    del candi_outputs
                    del item_outputs        
            self.all_embs = torch.cat(self.all_embs)
            
        u, seq, pos, neg, rank, shuffle, candi_set, files = data
        if shuffle:
            seqs = []
            for s in seq:
                non_zero_indices = np.nonzero(s)
                shuffled_non_zero_values = np.random.permutation(s[non_zero_indices])
                shuffled_arr = s.copy()
                shuffled_arr[non_zero_indices] = shuffled_non_zero_values

                seqs.append(shuffled_arr)
            seq = np.array(seqs)
            
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
        with torch.no_grad():
            for i in range(len(u)):
                log_emb_proj = self.log_emb_proj(log_emb[i]).unsqueeze(0)
                
                # log_emb_proj = F.normalize(self.log_emb_proj(log_emb[i]).unsqueeze(0),p=2,dim=1)
                
                text_input = []
                interact_embs = []
                candidate_embs = []
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                # interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= shuffle)
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= False, time = self.args.time)
                
                if self.args.load_candi:
                    candidate_ids = self.candi_set[u[i]]
                else:
                    candidate_num = 100
                    candidate_ids = self.make_candidate(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, candi_set)
                
                if self.args.baseline == 'tallrec':
                    input_text = ''
                else:
                    input_text = 'This is user representation from recommendation models : [UserRep]. '

                input_text += 'This user has made a series of purchases in the following order: '
                    
                input_text += interact_text

                input_text +='. Based on this sequence of purchases, to recommend one next item for this user, generate user representation token:[UserOut][UserOut][UserOut]'
                
                text_input.append(input_text)
                
                
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                

                max_input_length = 1024
                
                llm_tokens = self.llm.llm_tokenizer(
                    text_input,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=max_input_length,
                ).to(self.device)
                
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                
                if self.args.baseline == 'tallrec':
                    inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]'], embs= {})
                else:
                    inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[UserRep]', '[HistoryEmb]'], embs= {'[UserRep]':log_emb_proj, '[HistoryEmb]':interact_embs})

                with torch.amp.autocast(dev):
                    outputs = self.llm.llm_model.forward(
                        inputs_embeds=inputs_embeds,
                        # attention_mask=attention_mask,
                        # return_dict=True,
                        # labels=targets,
                        output_hidden_states=True
                    )
                    
                    indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                    # user_outputs = torch.stack([outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))]).squeeze(1)
                    user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                    item_outputs = self.all_embs[np.array(candidate_ids)-1]

                    user_outputs = self.llm.pred_user(user_outputs)
                    
                    logits= torch.mm(item_outputs, user_outputs.T).squeeze(-1)
                
                    logits = -1*logits
                    
                    rank = logits.argsort().argsort()[0].item()
                    
                    if rank < 10:
                        self.NDCG += 1 / np.log2(rank + 2)
                        self.HT += 1
                    self.users +=1
        return self.NDCG
    
    def generate_batch(self,data):
        if self.args.is_hpu:
            dev = 'hpu'
        else:
            dev = 'cuda'
        if self.all_embs == None:
            batch_ = 128
            if self.args.llm =='llama':
                batch_ = 64
            if self.args.rec_pre_trained_data == 'Electronics' or self.args.rec_pre_trained_data == 'Books':
                if self.args.llm =='llama':
                    batch_ = 32
                else:
                    batch_ = 64
            batches = self.split_into_batches(self.item_num, batch_)#128
            self.all_embs = []
            max_input_length = 1024
            for bat in tqdm(batches):
                candidate_text = []
                candidate_ids = []
                candidate_embs = []
                for neg_candidate in bat:
                    if self.args.baseline == 'tallrec':
                        candidate_text.append('The item title is as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + ", then generate item representation token:[ItemOut][ItemOut][ItemOut]")
                    else:
                        candidate_text.append('The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut][ItemOut][ItemOut]")
                    
                    candidate_ids.append(neg_candidate)
                with torch.no_grad():
                    candi_tokens = self.llm.llm_tokenizer(
                        candidate_text,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=max_input_length,
                    ).to(self.device)
                    # candidate_embs.append(self.item_emb_proj(self.standard(self.get_item_emb(candidate_ids))))
                    candidate_embs.append(self.item_emb_proj((self.get_item_emb(candidate_ids))))
                    # candidate_embs.append(F.normalize(self.item_emb_proj((self.get_item_emb(candidate_ids))), p=2,dim=1))

                    candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                    if self.args.baseline == 'tallrec':
                        candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemOut]'], embs= {})
                    else:
                        candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemOut]', '[HistoryEmb]'], embs= {'[HistoryEmb]':candidate_embs[0]})
                    with torch.amp.autocast(dev):
                        candi_outputs = self.llm.llm_model.forward(
                            inputs_embeds=candi_embeds,
                            # attention_mask=attention_mask,
                            # return_dict=True,
                            # labels=targets,
                            output_hidden_states=True
                        )
                        
                        indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]')
                        # item_outputs = torch.stack([candi_outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))]).squeeze(1)
                        item_outputs = torch.cat([candi_outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                        
                        item_outputs = self.llm.pred_item(item_outputs)
                    
                    self.all_embs.append(item_outputs)
                    del candi_outputs
                    del item_outputs        
            self.all_embs = torch.cat(self.all_embs)
            
        u, seq, pos, neg, rank, shuffle, candi_set, files = data
        
        original_seq = seq.copy()
        if shuffle:
            # seqs = []
            # for s in seq:
            #     non_zero_indices = np.nonzero(s)
            #     shuffled_non_zero_values = np.random.permutation(s[non_zero_indices])
            #     shuffled_arr = s.copy()
            #     shuffled_arr[non_zero_indices] = shuffled_non_zero_values

            #     seqs.append(shuffled_arr)
            # seq = np.array(seqs)
            
            seqs = []
            for s in seq:
                non_zero_indices = np.nonzero(s[-10:])
                shuffled_non_zero_values = np.random.permutation(s[-10:][non_zero_indices])
                shuffled_arr = s.copy()
                shuffled_arr[-10:][non_zero_indices] = shuffled_non_zero_values

                seqs.append(shuffled_arr)
            seq = np.array(seqs)
            
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            log_emb_proj = self.log_emb_proj(log_emb)
        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            for i in range(len(u)):
                # log_emb_proj = self.log_emb_proj(log_emb[i]).unsqueeze(0)
                
                # log_emb_proj = F.normalize(self.log_emb_proj(log_emb[i]).unsqueeze(0),p=2,dim=1)
                
                
                
                candidate_embs = []
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                # interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= shuffle)
                if not self.args.time:
                    interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= True, original_seq=original_seq[i][original_seq[i]>0], time = False)
                else:
                    interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= True, original_seq=original_seq[i][original_seq[i]>0])
                
                if self.args.load_candi:
                    candidate_ids = self.candi_set[u[i]]
                else:
                    candidate_num = 100
                    candidate_ids = self.make_candidate(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, candi_set)
                
                candidate.append(candidate_ids)
                
                if self.args.baseline == 'tallrec' or self.args.baseline =='llara':
                    input_text = ''
                else:
                    input_text = 'This is user representation from recommendation models : [UserRep]. '
                    

                input_text += 'This user has made a series of purchases in the following order: '
                    
                input_text += interact_text
                

                input_text +='. Based on this sequence of purchases, to recommend one next item for this user, generate user representation token:[UserOut][UserOut][UserOut]'
                
                text_input.append(input_text)
                
                # interact_embs.append(self.item_emb_proj(self.standard(self.get_item_emb(interact_ids))))
                # candidate_embs.append(self.item_emb_proj(self.standard(self.get_item_emb(candidate_ids))))
                
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                
                # interact_embs.append(F.normalize(self.item_emb_proj((self.get_item_emb(interact_ids))),p=2,dim=1))

            max_input_length = 1024
            
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)
            
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            
            if self.args.baseline == 'tallrec':
                inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]'], embs= {})
            elif self.args.baseline =='llara':
                inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]','[HistoryEmb]'], embs= {'[HistoryEmb]':interact_embs})
            else:
                inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[UserRep]', '[HistoryEmb]'], embs= {'[UserRep]':log_emb_proj, '[HistoryEmb]':interact_embs})

            with torch.amp.autocast(dev):
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,
                    # attention_mask=attention_mask,
                    # return_dict=True,
                    # labels=targets,
                    output_hidden_states=True
                )
                
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                # user_outputs = torch.stack([outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))]).squeeze(1)
                user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)
                
                for i in range(len(candidate)):
                    
                    item_outputs = self.all_embs[np.array(candidate[i])-1]
                    
                    logits= torch.mm(item_outputs, user_outputs[i].unsqueeze(0).T).squeeze(-1)
                
                    logits = -1*logits
                    
                    # rank = logits.argsort().argsort()[0].item()
                    
                    l_ = [i for i in range(len(logits))]
                    random.shuffle(l_)
                    
                    rank = logits[l_].argsort().argsort()[l_.index(0)].item()
                    
                    if rank < 10:
                        self.NDCG += 1 / np.log2(rank + 2)
                        self.HT += 1
                    if rank < 20:
                        self.NDCG_20 += 1 / np.log2(rank + 2)
                        self.HIT_20 += 1
                    self.users +=1
        return self.NDCG

    def extract_emb(self,data):    
        u, seq, pos, neg, original_seq,rank, shuffle, candi_set, files = data
        # original_seq = seq.copy()

            
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            log_emb_proj = self.log_emb_proj(log_emb)
        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            for i in range(len(u)):
                # log_emb_proj = self.log_emb_proj(log_emb[i]).unsqueeze(0)
                
                # log_emb_proj = F.normalize(self.log_emb_proj(log_emb[i]).unsqueeze(0),p=2,dim=1)
                
                
                
                # candidate_embs = []
                # target_item_id = pos[i]
                # target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                # interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= shuffle)
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= True,original_seq=original_seq[i][original_seq[i]>0],time = self.args.time)
                
                if self.args.baseline == 'tallrec' or self.args.baseline =='llara':
                    input_text = ''
                else:
                    input_text = 'This is user representation from recommendation models : [UserRep]. '
                     
                    #no user
                    # input_text = ''
                    

                input_text += 'This user has made a series of purchases in the following order: '
                    
                input_text += interact_text
                

                input_text +='. Based on this sequence of purchases, to recommend one next item for this user, generate user representation token:[UserOut][UserOut][UserOut]'
                
                text_input.append(input_text)
                
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                

            max_input_length = 1024
            
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)
            
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            
            if self.args.baseline == 'tallrec':
                inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]'], embs= {})
            elif self.args.baseline == 'llara':
                inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[HistoryEmb]'], embs= { '[HistoryEmb]':interact_embs})
            else:
                inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[UserRep]', '[HistoryEmb]'], embs= {'[UserRep]':log_emb_proj, '[HistoryEmb]':interact_embs})

                #no user
                # inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[HistoryEmb]'], embs= { '[HistoryEmb]':interact_embs})

            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,
                    # attention_mask=attention_mask,
                    # return_dict=True,
                    # labels=targets,
                    output_hidden_states=True
                )
                
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                # user_outputs = torch.stack([outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))]).squeeze(1)
                user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)
                
                self.extract_embs_list.append(user_outputs.detach().cpu())
                
        return 0

                
    def generate_all(self, data):#token 여러개 안해둠
        if self.all_embs == None:
            batch_ = 128
            if self.args.rec_pre_trained_data == 'Electronics' or self.args.rec_pre_trained_data == 'Books':
                batch_ = 64
            batches = self.split_into_batches(self.item_num, batch_)#128
            self.all_embs = []
            max_input_length = 1024
            for bat in tqdm(batches):
                candidate_text = []
                candidate_ids = []
                candidate_embs = []
                for neg_candidate in bat:
                    if self.args.task:
                        candidate_text.append('Task:[RecTask]. The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut][ItemOut][ItemOut]")
                        # candidate_text.append('Task:[ItemTask]. The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut]")
                    else:
                        candidate_text.append('The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut][ItemOut][ItemOut]")
                    
                    candidate_ids.append(neg_candidate)
                with torch.no_grad():
                    candi_tokens = self.llm.llm_tokenizer(
                        candidate_text,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=max_input_length,
                    ).to(self.device)
                    # candidate_embs.append(self.item_emb_proj(self.standard(self.get_item_emb(candidate_ids))))
                    candidate_embs.append(self.item_emb_proj((self.get_item_emb(candidate_ids))))
                    
                    # candidate_embs.append(F.normalize(self.item_emb_proj((self.get_item_emb(candidate_ids))), p=2, dim=1))

                    candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                    if self.args.task:
                        candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[RecTask]','[ItemOut]', '[HistoryEmb]'], embs= {'[HistoryEmb]':candidate_embs[0]})
                        # candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemTask]','[ItemOut]', '[HistoryEmb]'], embs= {'[HistoryEmb]':candidate_embs[0]})
                    else:
                        candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemOut]', '[HistoryEmb]'], embs= {'[HistoryEmb]':candidate_embs[0]})

                    with torch.amp.autocast('cuda'):
                        candi_outputs = self.llm.llm_model.forward(
                            inputs_embeds=candi_embeds,
                            # attention_mask=attention_mask,
                            # return_dict=True,
                            # labels=targets,
                            output_hidden_states=True
                        )
                        
                        indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]')
                        item_outputs = torch.stack([candi_outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))]).squeeze(1)
                        # indx = self.llm.get_embeddings(candi_tokens, '[ItemTask]')
                        # Taskoutputs = torch.cat([candi_outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))])
                        # item_outputs*=Taskoutputs
                        item_outputs = self.llm.pred_item(item_outputs)
                    
                    self.all_embs.append(item_outputs)
                    del candi_outputs
                    del item_outputs        
            self.all_embs = torch.cat(self.all_embs)
            
        u, seq, pos, neg, rank, shuffle, candi_set, files = data
        #Last Item modi
        # for s in seq:
        #     for i in range(len(s)-3):
        #         s[i] = 0
        
        if shuffle:
            seqs = []
            for s in seq:
                non_zero_indices = np.nonzero(s)
                shuffled_non_zero_values = np.random.permutation(s[non_zero_indices])
                shuffled_arr = s.copy()
                shuffled_arr[non_zero_indices] = shuffled_non_zero_values

                seqs.append(shuffled_arr)
            seq = np.array(seqs)
            
        
        
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
        with torch.no_grad():
            for i in range(len(u)):
                log_emb_proj = self.log_emb_proj(log_emb[i]).unsqueeze(0)
                
                text_input = []
                interact_embs = []
                candidate_embs = []
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= shuffle)
                
                if self.args.task:
                    input_text = 'Task:[RecTask]. This is user representation from recommendation models : [UserRep]. '
                else:
                    input_text = 'This is user representation from recommendation models : [UserRep]. '
                    
                # if self.args.rec_pre_trained_data == 'Movies_and_TV':
                #     input_text += 'This user has watched '
                # else:
                input_text += 'This user has made a series of purchases in the following order: '
                    
                input_text += interact_text
                
                # if self.args.rec_pre_trained_data == 'Movies_and_TV':
                #     input_text +=' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
                # else:
                input_text +='. Based on this sequence of purchases, to recommend one next item for this user, generate user representation token:[UserOut]'
                    # input_text +=". Generate user representation token:[UserOut], based on this user representation from recommendation model and sequence of purchases, to recommend one next item for this user."   

                # input_text += candidate_text
                # input_text += '. The recommendation is '
                
                text_input.append(input_text)
                
                # interact_embs.append(self.item_emb_proj(self.standard(self.get_item_emb(interact_ids))))
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                # interact_embs.append(F.normalize(self.item_emb_proj((self.get_item_emb(interact_ids))),p=2,dim=1))
                
                max_input_length = 1024
                
                llm_tokens = self.llm.llm_tokenizer(
                    text_input,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=max_input_length,
                ).to(self.device)
                
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                
                if self.args.task:
                    inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[RecTask]','[UserOut]', '[UserRep]', '[HistoryEmb]'], embs= {'[UserRep]':log_emb_proj, '[HistoryEmb]':interact_embs})
                else:
                    inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[UserRep]', '[HistoryEmb]'], embs= {'[UserRep]':log_emb_proj, '[HistoryEmb]':interact_embs})

                # inputs_embeds = self.llm.replace_out_token(llm_tokens, inputs_embeds, 'User')

                with torch.cuda.amp.autocast():
                    outputs = self.llm.llm_model.forward(
                        inputs_embeds=inputs_embeds,
                        # attention_mask=attention_mask,
                        # return_dict=True,
                        # labels=targets,
                        output_hidden_states=True
                    )
                    
                    indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                    user_outputs = torch.stack([outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))]).squeeze(1)

                    # indx = self.llm.get_embeddings(llm_tokens, '[RecTask]')
                    # Taskoutputs = torch.cat([outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))])
                    # user_outputs*=Taskoutputs
                    user_outputs = self.llm.pred_user(user_outputs)

                
                # logits = item_outputs.matmul(user_outputs.unsqueeze(-1)).squeeze(-1)
                
                logits= torch.mm(self.all_embs, user_outputs.T).squeeze(-1)
                
                logits = -1*logits
                
                rank = logits.argsort().argsort()[pos.item()-1].item()
                # rank = logits.argsort().argsort()[pos.item()].item()
                if rank < 10:
                    self.NDCG += 1 / np.log2(rank + 2)
                    self.HT += 1
                self.users +=1

        return self.NDCG