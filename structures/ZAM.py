import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from struct_functions import sequence_mask
from struct_functions import sequence_mask_lt
from struct_functions import L2Regularization

from HEM import HEM

class ZAM(HEM):
    def __init__(self, word_num, item_num, user_num, emb_dim, behavior_len, att_hidden_units_num, user_word_map, item_title_map, query_seg_map, user_click_map, user_word_lens, item_title_lens, query_seg_lens, user_click_lens, word_distribution, item_distribution, neg_sample_num, noise_rate, LAMBDA, L2_weight , score_func, device):
        super(ZAM,self).__init__(word_num, item_num, user_num, emb_dim, user_word_map, item_title_map, query_seg_map, user_word_lens, item_title_lens, query_seg_lens, word_distribution, item_distribution, neg_sample_num, noise_rate, LAMBDA, L2_weight, score_func, device)
        
        self.behavior_len = behavior_len
        self.att_hidden_units_num = att_hidden_units_num
        self.user_click_map = user_click_map.to(device)
        self.user_click_lens = user_click_lens.to(device)
    
        self.register_parameter(name='att_weight_h', param=nn.Parameter(torch.randn(att_hidden_units_num),requires_grad=True))
        self.register_parameter(name='att_weight_f', param=nn.Parameter(torch.randn(emb_dim, emb_dim, att_hidden_units_num),requires_grad=True))
        self.register_parameter(name='att_bias_f', param=nn.Parameter(torch.randn(emb_dim, att_hidden_units_num),requires_grad=True))
    
    def get_user_att_batch(self, batch_size, att_index_list, query_emb_batch, att_map, att_lens):
        att_ids = torch.index_select(att_map, 0, att_index_list).long().view(-1).to(self.device)
        
        # [batch_size X behavior_len X word_size]
        query_att_item_emb = torch.index_select(self.item_emb_with_zero, 0, att_ids).view(batch_size,self.behavior_len,self.emb_dim)

        # [batch_size, word_size] X [word_size X word_size X att_hidden_units_num] = [batch_size X word_size X att_hidden_units_num]
        t1 = torch.mm(query_emb_batch, self.att_weight_f.view(self.emb_dim,-1))

        t2 = t1.view(batch_size,self.emb_dim,self.att_hidden_units_num) + self.att_bias_f
        # [batch_size X behavior_len X word_size] X [batch_size X word_size X att_hidden_units_num] = [batch_size X behavior_len X att_hidden_units_num]
        t3 = torch.matmul(query_att_item_emb,torch.tanh(t2))
        
        # [batch_size X behavior_len X att_hidden_units_num] X [att_hidden_units_num] = [batch_size X behavior_len]
        t4 = torch.matmul(t3, self.att_weight_h)
        query_att_len_index = torch.index_select(att_lens, 0, att_index_list).long().view(-1)
        mask = sequence_mask(query_att_len_index, self.behavior_len)
        masked_item_weight = t4.masked_fill(mask, -1e9).view(batch_size, self.behavior_len, 1) 
        item_weight = F.softmax(masked_item_weight,1)

        user_att_batch = torch.sum(item_weight * query_att_item_emb, 1)

        return user_att_batch

    def forward(self, x):
        self.append_zero_vec()
        batch_size = x.size()[0]
        user_num = x[:,0].long()
        query_num = x[:,1].long()
        item_num = x[:,2].long()
        
        item_emb_batch = self.item_embedding(item_num)

        query_word_index_batch = torch.index_select(self.query_seg_map, 0, query_num).view(-1).long()
        query_word_batch = torch.index_select(self.word_emb_with_zero ,0 , query_word_index_batch).view(batch_size,-1,self.emb_dim)

        query_lens_batch = torch.index_select(self.query_seg_lens, 0, query_num).view(batch_size,1)

        query_emb_batch = torch.sum(query_word_batch,1) / query_lens_batch

        projected_query = self.query_project(query_emb_batch)
        
        user_emb_batch = self.get_user_att_batch(batch_size, user_num, query_emb_batch, self.user_click_map, self.user_click_lens)

        personalized_query = self.LAMBDA * projected_query + (1-self.LAMBDA) * user_emb_batch
        item_loss, i_embs = self.get_generation_loss(item_num, item_emb_batch, self.item_title_map, self.item_title_lens, self.word_distribution)
        
        uqi_loss, uqi_embs = self.get_uqi_loss(personalized_query, item_num, item_emb_batch, self.item_distribution)

        return uqi_loss + item_loss
    
    def all_item_test(self, u_id, q_id):
        self.append_zero_vec()
        query_word_index_batch = torch.index_select(self.query_seg_map, 0, q_id).view(-1).long()
        query_word_batch = torch.index_select(self.word_emb_with_zero ,0 , query_word_index_batch).view(1,-1,self.emb_dim)
        query_len = torch.index_select(self.query_seg_lens, 0, q_id).view(1, 1)
        query_emb = torch.sum(query_word_batch,1) / query_len
        user_emb = self.get_user_att_batch(1, u_id, query_emb, self.user_click_map, self.user_click_lens)

        projected_query = self.query_project(query_emb)
        personalized_query = self.LAMBDA * projected_query + (1-self.LAMBDA) * user_emb
        item_ids = torch.arange(0, self.item_embedding.weight.size()[0]).long().to(self.device)
        sim_score = self.get_sim_score(personalized_query, self.item_embedding.weight, item_ids, self.item_bias).view(-1)
        return sim_score
