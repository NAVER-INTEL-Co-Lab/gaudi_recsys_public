import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from title_generation_models.title_generation_llm4rec_baseline import *
# from title_generation_models.title_generation_llm4rec_baseline_testing import *

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
            # if args.pretrain_stage1:
            #     self.sbert = SentenceTransformer('nq-distilbert-base-v1')
            #     self.mlp2 = two_layer_mlp(self.sbert_dim)
                
        if args.pretrain_stage2 or args.inference or args.extract:
            
            if args.inference:
                epochs = args.infer_epoch
                out_dir = f'./title_generation_models/{args.save_dir}/{args.rec_pre_trained_data}_' + f'{args.llm}_{epochs}_'
                if self.args.baseline != 'a-llmrec' and self.args.baseline !='vanilla':
                    self.llm = llm4rec(device=self.device, llm_model=args.llm, load_lora=True, load_config=out_dir, args = self.args)
                else:
                    print(self.args.baseline, 'Load LLM')
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

            if self.args.baseline =='a-llmrec':
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

        out_dir = f'./title_generation_models/{args.save_dir}/'
        if best:
            out_dir = out_dir[:-1] + 'best/'
        
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_'
        if args.baseline =='a-llmrec':
            out_dir += f'{epoch1}_'
        out_dir += f'{args.llm}_{epoch2}_'
        if args.pretrain_stage2:
            if args.baseline != 'tallrec':
                torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
                torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            
            
            if self.args.baseline != 'a-llmrec':
                if self.args.is_hpu:
                    self.llm.llm_model.to('cpu')
                    self.llm.llm_model.save_pretrained(out_dir, save_embedding_layers=True)
                    self.llm.llm_model.to(args.device)
                else:
                    self.llm.llm_model.save_pretrained(out_dir, save_embedding_layers=True)
            # if self.args.baseline != 'a-llmrec':
            #     torch.save(self.llm.llm_model.base_model.model.model.embed_tokens.state_dict(), out_dir + 'token.pt')
            # else:
            #     torch.save(self.llm.llm_model.model.embed_tokens.state_dict(), out_dir + 'token.pt')
            
            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        # out_dir = f'./models/saved_models_new_loss_three_batch_infonce/{args.rec_pre_trained_data}_'
        # out_dir = f'./models/saved_models_new_loss_three_infonce_mse/{args.rec_pre_trained_data}_'
        # out_dir = f'./models/saved_models_original_raw_video_re/{args.rec_pre_trained_data}_'
        if args.baseline =='a-llmrec' and args.pretrain_stage2:
            out_dir = f'./title_generation_models/{args.save_dir}best/{args.rec_pre_trained_data}_'
        else:
            out_dir = f'./title_generation_models/{args.save_dir}/{args.rec_pre_trained_data}_'
        if args.baseline =='a-llmrec':
            out_dir += f'{phase1_epoch}_'
            mlp = torch.load(out_dir + 'mlp.pt', map_location = args.device)
            self.mlp.load_state_dict(mlp)
            del mlp
            for name, param in self.mlp.named_parameters():
                param.requires_grad = False
                
        if args.inference:
            out_dir += f'{args.llm}_{phase2_epoch}_'
            
            if (args.baseline != 'tallrec') and (args.baseline != 'vanilla'):
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

        l = [datetime.utcfromtimestamp(int(self.text_name_dict['time'][i][user])/1000) for i in item]
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
                if self.args.baseline == 'tallrec' or (self.args.baseline =='llara' and llara_text) or self.args.baseline =='vanilla':
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
                if self.args.baseline == 'tallrec' or (self.args.baseline =='llara' and llara_text) or self.args.baseline =='vanilla':
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
            candidate_text = ['"Title"' + target_item_title]
        else:
            candidate_text = ['"Title"' + target_item_title + "[CandidateEmb]"]


        for neg_candidate in neg_item_id[:candidate_num - 1]:
            if self.args.baseline == 'tallrec' or (self.args.baseline =='llara' and llara_text):
                candidate_text.append('"Title"' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False))
            else:
                candidate_text.append('"Title"' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[CandidateEmb]")

            candidate_ids.append(neg_candidate)
        
        
        random_ = np.random.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        candidate_ids = np.array(candidate_ids)[random_]
        target_number = [f'Target No.{i+1};' for i in range(len(candidate_text))]
        
        answer = target_number[np.where(random_ == 0)[0][0]]
        
        new_list = [target_number[i] + candidate_text[i] for i in range(len(candidate_text))]
        
        return ','.join(candidate_text), candidate_ids
        # return ','.join(new_list), candidate_ids, answer
    
    
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
    
    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        print(self.args.save_dir, self.args.rec_pre_trained_data)
        optimizer.zero_grad()
        u, seq, pos, neg = data
        
        mean_loss = 0
        
        text_input = []
        text_output = []

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
            
            if not self.args.time:
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], llara_text=llara_text, time = False)
            else:
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], llara_text=llara_text)
            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, llara_text=llara_text)
            # candidate_text, candidate_ids, ans = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, llara_text=llara_text)

            if self.args.baseline == 'llara':
                for __ in range(candidate_num):
                    llara_ind_item.append(1)
            
            if self.args.baseline == 'tallrec' or (self.args.baseline =='llara'):
                input_text = ''
            else:
                input_text = 'This is user representation from recommendation models : [UserRep]. '

            input_text += 'This user has made a series of purchases in the following order: '
                
            input_text += interact_text
            
            input_text +=' in the previous. \n Chose one "Title" to recommend for this user to buy next from the following item "Title" set, '

            input_text += candidate_text
            input_text += '. The "Title" is '

            text_input.append(input_text)
            text_output.append(target_item_title)
            # text_output.append(ans)

            candidates_pos += candidate_text             
            
            
            interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
            candidate_embs.append(self.item_emb_proj((self.get_item_emb([candidate_ids]))).squeeze(0))
                        
                
        log_emb = self.log_emb_proj(log_emb)
        
        if self.args.baseline != 'llara':
            llara_ind = None
            llara_ind_item = None
        
        samples = {'text_input': text_input, 'text_output': text_output, 'log_emb':log_emb, 'interact': interact_embs, 'candidate_embs':candidate_embs, 'llara_ind': llara_ind, 'llara_ind_item':llara_ind_item}
        
        

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
            
        u, seq, pos, neg, rank, shuffle, candi_set, files = data

        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []

        if shuffle:
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

        with torch.no_grad():
            for i in range(len(u)):
                
                # log_emb_proj = F.normalize(self.log_emb_proj(log_emb[i]).unsqueeze(0),p=2,dim=1)
                
                
                
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                # interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= shuffle)
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= False, time = self.args.time)
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)
                # candidate_text, candidate_ids, ans = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)

                
                if self.args.baseline == 'tallrec' or self.args.baseline == 'llara' or self.args.baseline =='vanilla':
                    input_text = ''
                else:
                    input_text = 'This is user representation from recommendation models : [UserRep]. '

                input_text += 'This user has made a series of purchases in the following order: '
                    
                input_text += interact_text
                
                input_text +=' in the previous. \n Chose one "Title" to recommend for this user to buy next from the following item "Title" set, '

                input_text += candidate_text
                input_text += '. The "Title" is '
                answer.append(target_item_title)
                # answer.append(ans)

                text_input.append(input_text)
                
                
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
        
        log_emb = self.log_emb_proj(log_emb)

        with torch.no_grad():
            self.llm.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                
                if (not self.args.baseline =='tallrec') and (not self.args.baseline =='llara'):
                    inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]'], embs = {'[UserRep]':log_emb, '[HistoryEmb]':interact_embs,'[CandidateEmb]': candidate_embs})
                if not self.args.baseline =='tallrec' and self.args.baseline =='llara':
                    inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = [ '[HistoryEmb]', '[CandidateEmb]'], embs = { '[HistoryEmb]':interact_embs,'[CandidateEmb]': candidate_embs})

                attention_mask = llm_tokens.attention_mask
                    
                outputs = self.llm.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    top_p=0.5,#0.1
                    # temperature=0.9,#0
                    num_beams=3,#1
                    # top_p=0.1,#0.1
                    temperature=0.0,#0
                    # num_beams=1,#1
                    max_new_tokens=8,
                    min_length=1,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                    pad_token_id=self.llm.llm_tokenizer.eos_token_id
                )

            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

        for i in range(len(text_input)):

            f = open(f'./title_generation_models/{self.args.save_dir}/{self.args.rec_pre_trained_data}_recommendation_output_{self.args.shuffle}.txt','a')
            f.write(text_input[i])
            f.write('\n\n')
            
            f.write('Answer: '+ answer[i])
            f.write('\n\n')
            
            f.write('LLM: '+str(output_text[i]))
            f.write('\n\n')
            f.close()

        return self.NDCG
    

    def extract_emb(self,data):    
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
                
                
                
                # candidate_embs = []
                # target_item_id = pos[i]
                # target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                # interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= shuffle)
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i], shuffle= True,original_seq=original_seq[i][original_seq[i]>0], time = self.args.time)
                
                if self.args.baseline == 'tallrec':
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

                
    