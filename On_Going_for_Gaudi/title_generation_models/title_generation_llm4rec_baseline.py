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

        if self.args.baseline != 'a-llmrec' and self.args.baseline !='vanilla':
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
                            user_vector = torch.cat((user_vector [:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
                        elif 'ItemOut' in t:
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
                            user_vector = torch.cat((user_vector [:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
                        elif 'ItemOut' in t:
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

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len



    
    def forward(self, samples, mode = 0):
        if mode ==0:
            return self.train_mode0(samples)


    def train_mode0(self,samples):
        max_input_length = 1024
        log_emb = samples['log_emb']

        # atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        # atts_llm = atts_llm.unsqueeze(1)
        

        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.device)
        
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )
        
        targets = llm_tokens['input_ids'].masked_fill(llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100)

        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # empty_targets = (torch.ones(atts_llm.size(), dtype=torch.long).to(self.device).fill_(-100))

        # targets = torch.cat([empty_targets, targets], dim=1)


        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        
        # print(log_emb.unsqueeze(0).shape)
        if (not self.args.baseline =='tallrec' ) and (not self.args.baseline == 'llara'):
            inputs_embeds = self.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]'], embs= {'[UserRep]':log_emb, '[HistoryEmb]':samples['interact'],'[CandidateEmb]':samples['candidate_embs']} ,llara_ind=samples['llara_ind'])
        if (not self.args.baseline =='tallrec' ) and self.args.baseline =='llara':
            inputs_embeds = self.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[HistoryEmb]', '[CandidateEmb]'], embs= {'[HistoryEmb]':samples['interact'],'[CandidateEmb]':samples['candidate_embs']} ,llara_ind=samples['llara_ind'])

        attention_mask = llm_tokens['attention_mask']

        # log_emb = log_emb.unsqueeze(1)
        # inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
        attention_mask = llm_tokens['attention_mask']
        # attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with torch.cuda.amp.autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return loss
    