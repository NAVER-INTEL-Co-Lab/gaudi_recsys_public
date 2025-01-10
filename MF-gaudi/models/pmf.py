import numpy as np
import torch
import torch.nn as nn

class PMF(torch.nn.Module):
    def __init__(self, user_num, item_num, label_matrix, args,lam_u = 0.3, lam_v = 0.3):
        super(PMF, self).__init__()
        
        self.kwargs = {'user_num': user_num, 'item_num':item_num, 'args':args}
        self.user_num = user_num
        self.item_num = item_num
        self.label_matrix = label_matrix
        
        self.lam_u = lam_u
        self.lam_v = lam_v
        self.act = nn.Sigmoid()
        
        self.dev = args.device
        self.nn_parameter = args.nn_parameter
        
        if self.nn_parameter:
            self.user_emb = nn.Parameter(torch.normal(0,0.1, size=(self.user_num, args.hidden_units)))
            self.item_emb = nn.Parameter(torch.normal(0,0.1, size=(self.user_num, args.hidden_units)))
        else:
            self.user_emb = nn.Embedding(self.user_num, args.hidden_units, padding_idx=0)
            self.item_emb = nn.Embedding(self.item_num, args.hidden_units, padding_idx=0)
            
    def forward(self, user_ids, item_ids):
        
        
        if self.nn_parameter:
            user_embs = self.user_emb[torch.LongTensor(user_ids).to(self.dev)]
            item_embs = self.item_emb[torch.LongTensor(item_ids).to(self.dev)]
        else:
            user_embs = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
            item_embs = self.item_emb(torch.LongTensor(item_ids).to(self.dev))
            
        logits = self.act(torch.matmul(user_embs, item_embs.T))
        labels = self.label_matrix[user_ids,:][:,item_ids]

        loss = ((labels-logits)**2).mean()
        # loss += self.lam_u * torch.sum(user_embs.norm(dim=1))
        # loss += self.lam_v * torch.sum(item_embs.norm(dim=1))
        
        return loss
    
    def predict(self, user_ids, item_indices):
        
        if self.nn_parameter:
            user_embs = self.user_emb[torch.LongTensor(user_ids).to(self.dev)]
            item_embs = self.item_emb[torch.LongTensor(item_indices).to(self.dev)]
        else:
            user_embs = self.user_emb(torch.LongTensor(user_ids).to(self.dev))
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
            
        logits = self.act(torch.matmul(user_embs, item_embs.T))
        
        logits = logits.diag()

        total_len = len(logits)
        acc = torch.sum(logits>0.5).item()
        
        return [acc, total_len]