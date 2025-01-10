import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PeftConfig,
    PeftModel
)
class llm4rec(nn.Module):
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q_proj","v_proj"],
        load_lora = False,
        load_config = '',
        args= None
    ):
        super().__init__()
        self.device = device
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.args = args
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        if llm_model == 'llama':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif llm_model == 'llama-3b':
            model_id="meta-llama/Llama-3.2-3B-Instruct"
        else:
            raise Exception(f'{llm_model} is not supported')
        print()
        print("=========")
        # if self.device.type =='hpu':
        if self.device =='aaa':
            self.llm_model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16, device_map=self.device)
        else:
            # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            
            # self.llm_model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16, load_in_8bit=True, device_map=self.device)
            if self.args.is_hpu == True:
                self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.float16,)
            else:
                self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.float16,load_in_8bit=True,)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            
        
            
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[UserRep]','[HistoryEmb]','[CandidateEmb]', '[UserOut]', '[ItemOut]']})
        self.llm_tokenizer.add_special_tokens({'cls_token': "[CLS]"})
        
        
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        self.llm_model = prepare_model_for_kbit_training(self.llm_model)
        
        for _, param in self.llm_model.named_parameters():
            # if 'token' in _:
            #     param.requires_grad = True
            # else:
            #     param.requires_grad = False
            param.requires_grad = False

        if self.args.baseline != 'a-llmrec':
            if load_lora == False:
                self.llm_model = prepare_model_for_kbit_training(self.llm_model)
                self.llm_model = get_peft_model(self.llm_model, config)
                self.llm_model.print_trainable_parameters()
            
            if load_lora ==True:
                config = PeftConfig.from_pretrained(load_config)
                print(config)
                print("")
                print("")
                print("Load Config: ", load_config)
                print("")
                print("")
                self.llm_model = PeftModel.from_pretrained(self.llm_model, load_config, device_map = self.device)
                self.llm_model.to(self.device)
        
        
        self.pred_user = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),
                nn.LayerNorm(2048),
                # nn.GELU(),
                # nn.BatchNorm1d(2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 128)
            )
        nn.init.xavier_normal_(self.pred_user[0].weight)
        nn.init.xavier_normal_(self.pred_user[3].weight)
        
        
        self.pred_item = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),
                nn.LayerNorm(2048),
                # nn.GELU(),
                # nn.BatchNorm1d(2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 128)
            )
        nn.init.xavier_normal_(self.pred_item[0].weight)
        nn.init.xavier_normal_(self.pred_item[3].weight)
        
        if self.args.nn_parameter:
            self.CLS = nn.Parameter(torch.normal(self.llm_model.base_model.model.model.embed_tokens.weight.mean(),self.llm_model.base_model.model.model.embed_tokens.weight.std(), size = (1,self.llm_model.config.hidden_size)))
            self.CLS_item = nn.Parameter(torch.normal(self.llm_model.base_model.model.model.embed_tokens.weight.mean(),self.llm_model.base_model.model.model.embed_tokens.weight.std(), size = (1,self.llm_model.config.hidden_size)))
        else:
            
            if self.args.baseline !='a-llmrec':
                self.CLS = nn.Embedding(1,self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(self.CLS.weight, mean = self.llm_model.base_model.model.model.embed_tokens.weight.mean(), std = self.llm_model.base_model.model.model.embed_tokens.weight.std())
                self.CLS_item = nn.Embedding(1,self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(self.CLS_item.weight, mean = self.llm_model.base_model.model.model.embed_tokens.weight.mean(), std = self.llm_model.base_model.model.model.embed_tokens.weight.std())
            elif self.args.baseline == 'a-llmrec':
                self.CLS = nn.Embedding(1,self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(self.CLS.weight, mean = self.llm_model.model.embed_tokens.weight.mean(), std = self.llm_model.model.embed_tokens.weight.std())
                self.CLS_item = nn.Embedding(1,self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(self.CLS_item.weight, mean = self.llm_model.model.embed_tokens.weight.mean(), std = self.llm_model.model.embed_tokens.weight.std())
        

        self.mse = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.mse_info = nn.MSELoss(reduction='none')
        
        self.max_output_txt_len = max_output_txt_len

    
    def rec_loss(self,anchor, items, temperature=0.07):
        
        
        logits = torch.bmm(items.view(anchor.shape[0], -1, anchor.shape[1]), anchor.unsqueeze(2)).squeeze(2)
        # logits /= temperature
        
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    

    
    def replace_out_token_all(self, llm_tokens, inputs_embeds, token = [], embs= None, llara_ind = None):
        # inputs_embeds = inputs_embeds.clone()
        for t in token:
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            for inx in range(len(llm_tokens["input_ids"])):
                if llara_ind != None and self.args.baseline == 'llara' and not ('UserOut' in t or 'ItemOut' in t):
                    if llara_ind[inx] == 1:
                        user_vector = inputs_embeds[inx]
                        vectors.append(user_vector.unsqueeze(0))
                        continue
                idx_tensor=(llm_tokens["input_ids"][inx]==token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]
                
                if 'Emb' in t:
                    ee = embs[t][inx]
                    for idx, item_emb in zip(idx_tensor, ee):
                        # inputs_embeds[inx][idx]=item_emb
                        
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                
                elif 'Rep' in t:
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                else:
                    for idx in idx_tensor:
                        # if 'UserOut1' in t:
                        #     inputs_embeds[inx][idx] = self.Pred_CF(torch.tensor([0]).to(self.device))
                        if 'UserOut' in t:
                            if self.args.nn_parameter:
                                user_vector = torch.cat((user_vector [:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector [idx+1:]), dim=0)
                            else:
                                user_vector = torch.cat((user_vector [:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
                        elif 'ItemOut' in t:
                            if self.args.nn_parameter:
                                user_vector = torch.cat((user_vector [:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector [idx+1:]), dim=0)
                            else:
                                user_vector = torch.cat((user_vector [:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
                        
                vectors.append(user_vector.unsqueeze(0))
            inputs_embeds = torch.cat(vectors)        
        return inputs_embeds
    
    def replace_out_token_all_infer(self, llm_tokens, inputs_embeds, token = [], embs= None, llara_ind = None):
        # inputs_embeds = inputs_embeds.clone()
        for t in token:
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            for inx in range(len(llm_tokens["input_ids"])):
                if llara_ind != None and self.args.baseline == 'llara' and not ('UserOut' in t or 'ItemOut' in t):
                    if llara_ind[inx] == 1:
                        user_vector = inputs_embeds[inx]
                        vectors.append(user_vector.unsqueeze(0))
                        continue
                idx_tensor=(llm_tokens["input_ids"][inx]==token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]
                if 'Emb' in t:
                    ee = [embs[t][inx]]
                    # ee = embs[t][inx]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                                                
                elif 'Rep' in t:
                    for idx in idx_tensor:                        
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                else:
                    for idx in idx_tensor:
                        # if 'UserOut1' in t:
                        #     inputs_embeds[inx][idx] = self.Pred_CF(torch.tensor([0]).to(self.device))
                        if 'UserOut' in t:
                            if self.args.nn_parameter:
                                user_vector = torch.cat((user_vector [:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector [idx+1:]), dim=0)
                            else:
                                user_vector = torch.cat((user_vector [:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
                        elif 'ItemOut' in t:
                            if self.args.nn_parameter:
                                user_vector = torch.cat((user_vector [:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector [idx+1:]), dim=0)
                            else:
                                user_vector = torch.cat((user_vector [:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
                        
                vectors.append(user_vector.unsqueeze(0))
            inputs_embeds = torch.cat(vectors)        
        return inputs_embeds
        
    def get_embeddings(self, llm_tokens, token):
        token_idx = []
        token_id = self.llm_tokenizer(token, return_tensors="pt", add_special_tokens=False).input_ids.item()
        for inx in range(len(llm_tokens['input_ids'])):
            idx_tensor = (llm_tokens['input_ids'][inx] == token_id).nonzero().view(-1)
            token_idx.append(idx_tensor)
        return token_idx


    
    def forward(self, samples, mode = 0):
        if mode ==0:
            return self.train_mode0(samples)
        
        elif mode == 1:
            return self.train_mode1(samples)
        elif mode==5:
            return self.train_mode5(samples)


    def train_mode0(self,samples):
        max_input_length = 1024
        log_emb = samples['log_emb']
        llm_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        
        
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        # print(log_emb.unsqueeze(0).shape)
        
        if self.args.baseline =='tallrec':
            inputs_embeds = self.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]'], embs= {})
        elif self.args.baseline =='llara':
            inputs_embeds = self.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[HistoryEmb]'], embs= { '[HistoryEmb]':samples['interact']}, llara_ind=samples['llara_ind'])
        else:
            inputs_embeds = self.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[UserRep]', '[HistoryEmb]'], embs= {'[UserRep]':log_emb, '[HistoryEmb]':samples['interact']}, llara_ind=samples['llara_ind'])
        # inputs_embeds = self.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[RecTask]','[UserOut]', '[UserRep]', '[HistoryEmb]'], embs= {'[UserRep]':log_emb, '[HistoryEmb]':samples['interact']})
        
        candi_tokens = self.llm_tokenizer(
                samples['candidates_pos'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

        candi_embeds = self.llm_model.get_input_embeddings()(candi_tokens['input_ids'])

        if self.args.baseline =='tallrec':
            candi_embeds = self.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemOut]'], embs= {})
        else:
            candi_embeds = self.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemOut]', '[HistoryEmb]'], embs= {'[HistoryEmb]':samples['candidate_embs']},llara_ind=samples['llara_ind_item'])
        # candi_embeds = self.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[RecTask]','[ItemOut]', '[HistoryEmb]'], embs= {'[HistoryEmb]':samples['candidate_embs']})
        
        # inputs_embeds = self.replace_out_token(llm_tokens, inputs_embeds, 'User')
        
        # llm_tokens, inputs_embeds = self.replace_user_emb_token(llm_tokens, inputs_embeds, embs = log_emb)
        
        # llm_tokens, inputs_embeds = self.replace_hist_candi_token(llm_tokens, inputs_embeds, embs = samples['interact'], mode=0)
        
        # inputs_embeds = self.concat_CLS(inputs_embeds)
                
        # if self.device.type == 'hpu':
        if self.args.is_hpu:
            dev = 'hpu'
        else:
            dev = 'cuda'
        # with torch.amp.autocast('cuda'):
        with torch.amp.autocast(dev):
            # if self.device == 'adfa':
            #     with torch.autocast(device_type="hpu"):
            #         outputs = self.llm_model(
            #             inputs_embeds=inputs_embeds,
            #             # attention_mask=attention_mask,
            #             return_dict=True,
            #             # labels=targets,
            #         )
            # else:
            
            
            candi_outputs = self.llm_model.forward(
                inputs_embeds=candi_embeds,
                # attention_mask=attention_mask,
                # return_dict=True,
                # labels=targets,
                output_hidden_states=True
            )
            
            indx = self.get_embeddings(candi_tokens, '[ItemOut]')
            # item_outputs = torch.cat([candi_outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))])
            item_outputs = torch.cat([candi_outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
            
            # indx = self.get_embeddings(candi_tokens, '[RecTask]')
            # Taskoutputs = torch.cat([candi_outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))])
            # item_outputs*=Taskoutputs
            
            outputs = self.llm_model.forward(
                inputs_embeds=inputs_embeds,
                # attention_mask=attention_mask,
                # return_dict=True,
                # labels=targets,
                output_hidden_states=True
            )
            #'[UserOut]', '[ItemOut]'
            
            indx = self.get_embeddings(llm_tokens, '[UserOut]')
            # user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))])
            user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                
            # indx = self.get_embeddings(llm_tokens, '[RecTask]')
            # Taskoutputs = torch.cat([outputs.hidden_states[-1][i,indx[i]] for i in range(len(indx))])
            # user_outputs *=Taskoutputs
                
                
            

            
        
        user_outputs = self.pred_user(user_outputs)
        item_outputs = self.pred_item(item_outputs)
        
        loss = self.rec_loss(user_outputs, item_outputs)
        
        
        # pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)
        # loss = self.bce_criterion(pos_logits, pos_labels)
        # loss += self.bce_criterion(neg_logits, neg_labels)
        return loss
    