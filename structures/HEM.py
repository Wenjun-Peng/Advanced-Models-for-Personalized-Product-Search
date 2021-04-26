import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from struct_functions import sequence_mask
from struct_functions import sequence_mask_lt
from struct_functions import L2Regularization

class HEM(nn.Module):
    def __init__(self, word_num, item_num, user_num, emb_dim, user_word_map, item_title_map, query_seg_map, user_word_lens, item_title_lens, query_seg_lens, word_distribution, item_distribution, neg_sample_num, noise_rate, LAMBDA, L2_weight, score_func, device):
        super(HEM, self).__init__()
        
        # hyper parameters
        self.word_num = word_num
        self.user_num = user_num
        self.item_num = item_num
        self.emb_dim = emb_dim
        self.LAMBDA = LAMBDA
        self.user_word_map = user_word_map.to(device)
        self.query_seg_map = query_seg_map.to(device)
        self.item_title_map = item_title_map.to(device)
        self.user_word_lens = user_word_lens.to(device)
        self.item_title_lens = item_title_lens.to(device)
        self.query_seg_lens = query_seg_lens.to(device)
        self.word_distribution = word_distribution.to(device)
        self.item_distribution = item_distribution.to(device)
        self.neg_sample_num = neg_sample_num
        self.noise_rate = noise_rate
        self.L2_weight = L2_weight
        self.score_func = score_func
        self.device = device

        # layers of net
        self.word_embedding = nn.Embedding(
            num_embeddings =  word_num,
            embedding_dim = emb_dim,
        )
        
        self.user_embedding = nn.Embedding(
            num_embeddings =  user_num,
            embedding_dim = emb_dim,
        )
        
        self.item_embedding = nn.Embedding(
            num_embeddings =  item_num,
            embedding_dim = emb_dim,
        )
        
        
        self.query_project = nn.Sequential(
            nn.Linear(
                in_features = emb_dim,
                out_features = emb_dim,
            ),
            nn.Tanh()
        )
        
        if self.score_func == "bias_product":
            self.register_parameter(name='word_bias', param=nn.Parameter(torch.randn(self.word_num),requires_grad=True))
            self.register_parameter(name='item_bias', param=nn.Parameter(torch.randn(self.item_num),requires_grad=True))
        
        self.BCELoss_fun = nn.BCEWithLogitsLoss()
        
    def append_zero_vec(self):
        self.word_emb_with_zero = torch.cat([self.word_embedding.weight,torch.zeros(1,self.emb_dim).to(self.device)],0)
        self.item_emb_with_zero = torch.cat([self.item_embedding.weight,torch.zeros(1,self.emb_dim).to(self.device)],0)
        if self.score_func == "bias_product":
            self.word_bias_with_zero = torch.cat([self.word_bias,torch.zeros(1).to(self.device)],0)
            self.item_bias_with_zero = torch.cat([self.item_bias,torch.zeros(1).to(self.device)],0)
    
    def get_generation_loss(self, example_ids, example_emb_batch, example_word_map, example_lens, cadidate_distribution):
        batch_size = example_emb_batch.size()[0]
        
        example_word_index_batch = torch.index_select(example_word_map, 0, example_ids).view(-1).long()
        example_word_batch = torch.index_select(self.word_emb_with_zero, 0, example_word_index_batch).view(batch_size,-1,self.emb_dim)
        example_lens_batch = torch.index_select(example_lens, 0, example_ids)
        example_emb_batch = example_emb_batch.view(batch_size,-1,self.emb_dim)

        if self.score_func == "bias_product":
            example_word_bias_batch = torch.index_select(self.word_bias_with_zero, 0, example_word_index_batch).view(batch_size,-1)
            sim_score = torch.sum(example_emb_batch * example_word_batch,2) + example_word_bias_batch
        else:
            sim_score = torch.sum(example_emb_batch * example_word_batch,2)
        
        mask_weight = sequence_mask_lt(example_lens_batch,sim_score.size()[1]).float()

        sim_score = sim_score.view(-1)
        loss_func = nn.BCEWithLogitsLoss(weight=mask_weight.view(-1))
        pos_loss = loss_func(sim_score, torch.ones_like(sim_score).float().to(self.device))
        
        noise_ditribution = cadidate_distribution ** self.noise_rate
        
        noise_ids = torch.LongTensor(list(Data.WeightedRandomSampler(noise_ditribution, example_word_index_batch.size()[0]*self.neg_sample_num, True))).to(self.device)
        noise_word_batch = torch.index_select(self.word_embedding.weight, 0 ,noise_ids).view(batch_size, -1, self.emb_dim)
        
        if self.score_func == "bias_product":
            noise_word_bias_batch = torch.index_select(self.word_bias_with_zero, 0, noise_ids).view(batch_size,-1)
            noise_sim_score = torch.sum(example_emb_batch * noise_word_batch,2) + noise_word_bias_batch
        else:
            noise_sim_score = torch.sum(example_emb_batch * noise_word_batch,2)
        
        noise_lens_batch = example_lens_batch*self.neg_sample_num
        noise_mask_weight = sequence_mask_lt(noise_lens_batch, noise_sim_score.size()[1]).float().view(batch_size, -1, 1)
        
        noise_sim_score = noise_sim_score.view(-1)
        noise_loss_func = nn.BCEWithLogitsLoss(weight=noise_mask_weight.view(-1))
        neg_loss = noise_loss_func(noise_sim_score, torch.zeros_like(noise_sim_score).float().to(self.device))

        return pos_loss+neg_loss, [example_emb_batch, example_word_batch, noise_word_batch*noise_mask_weight]
    
    def get_uqi_loss(self, personalized_query, item_ids, item_emb_batch, item_distribution):
        batch_size = item_emb_batch.size()[0]
        
        if self.score_func == "bias_product":
            item_bias_batch = torch.index_select(self.item_bias_with_zero, 0 ,item_ids).view(batch_size,-1)
            sim_score = torch.sum(personalized_query * item_emb_batch,1) + item_bias_batch
        else:
            sim_score = torch.sum(personalized_query * item_emb_batch,1)
            
        sim_score = sim_score.view(-1) 
        
        pos_loss = self.BCELoss_fun(sim_score, torch.ones_like(sim_score))
        
        noise_ditribution = item_distribution ** self.noise_rate
        noise_ids = torch.LongTensor(list(Data.WeightedRandomSampler(noise_ditribution, batch_size*self.neg_sample_num, True))).to(self.device)
        noise_item_batch = torch.index_select(self.item_embedding.weight, 0, noise_ids).view(batch_size,-1,self.emb_dim)
        
        if self.score_func == "bias_product":
            noise_item_bias_batch = torch.index_select(self.item_bias_with_zero, 0 ,noise_ids).view(batch_size,-1)
            noise_sim_score = torch.sum(personalized_query.view(batch_size,-1,self.emb_dim) * noise_item_batch,2) + noise_item_bias_batch
        else:
            noise_sim_score = torch.sum(personalized_query.view(batch_size,-1,self.emb_dim) * noise_item_batch,2)
        
        noise_sim_score = noise_sim_score.view(-1)
        neg_loss = self.BCELoss_fun(noise_sim_score, torch.zeros_like(noise_sim_score).float().to(self.device))
        
        return pos_loss+neg_loss, [personalized_query, noise_item_batch]
    
    
    def forward(self, x):
        self.append_zero_vec()
        batch_size = x.size()[0]
        user_num = x[:,0].long()
        query_num = x[:,1].long()
        item_num = x[:,2].long()
        
        user_emb_batch = self.user_embedding(user_num)
        item_emb_batch = self.item_embedding(item_num)

        query_word_index_batch = torch.index_select(self.query_seg_map, 0, query_num).view(-1).long()

        query_word_batch = torch.index_select(self.word_emb_with_zero ,0 , query_word_index_batch).view(batch_size,-1,self.emb_dim)

        query_lens_batch = torch.index_select(self.query_seg_lens, 0, query_num).view(batch_size, 1)
        query_emb_batch = torch.sum(query_word_batch,1) / query_lens_batch

        projected_query = self.query_project(query_emb_batch)
        
        personalized_query = self.LAMBDA * projected_query + (1-self.LAMBDA) * user_emb_batch

        user_loss, u_embs = self.get_generation_loss(user_num, user_emb_batch, self.user_word_map, self.user_word_lens, self.word_distribution)
        item_loss, i_embs = self.get_generation_loss(item_num, item_emb_batch, self.item_title_map, self.item_title_lens, self.word_distribution)
        
        uqi_loss, uqi_embs = self.get_uqi_loss(personalized_query, item_num, item_emb_batch, self.item_distribution)
        
        # u_L2 = L2Regularization(self.L2_weight,u_embs)
        # i_L2 = L2Regularization(self.L2_weight,i_embs)
        # uqi_L2 = L2Regularization(self.L2_weight,uqi_embs)

        return user_loss+item_loss+uqi_loss
    
    def all_item_test(self, u_id, q_id):
        self.append_zero_vec()
        user_emb = self.user_embedding(u_id)
        query_word_index_batch = torch.index_select(self.query_seg_map, 0, q_id).view(-1).long()
        query_word_batch = torch.index_select(self.word_emb_with_zero ,0 , query_word_index_batch).view(1,-1,self.emb_dim)
        query_len = torch.index_select(self.query_seg_lens, 0, q_id).view(1, 1)
        query_emb = torch.sum(query_word_batch,1).view(1,self.emb_dim) / query_len
        projected_query = self.query_project(query_emb)
        personalized_query = self.LAMBDA * projected_query + (1-self.LAMBDA) * user_emb
        
        sim_score = torch.sum(personalized_query*self.item_embedding.weight,1)
        if self.score_func == "bias_product":
            sim_score = sim_score + self.item_bias
        return sim_score
