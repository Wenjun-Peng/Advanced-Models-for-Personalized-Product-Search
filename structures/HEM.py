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

    def get_sim_score(self, X, Y, bias_ids=None, bias_map=None):
        '''
        :param X: batch of query, item or user
        :param Y: batch of item or word
        :param Y_ids: indices of Y
        :param X_map: word_ids or item_ids corresponding to Y
        :param bias_map: bias corresponding to word_ids or item_ids
        :return: score of (X,Y)
        '''

        X_batch_size = X.size()[0]
        Y_batch_size = Y.size()[0]
        X = X.view(X_batch_size, -1, self.emb_dim)
        Y = Y.view(Y_batch_size, -1, self.emb_dim)
        if self.score_func == "cosine":
            X_for_cos = X / torch.sqrt(torch.sum(X*X, -1), -1)
            Y_for_cos = Y / torch.sqrt(torch.sum(Y*Y, -1), -1)
            return torch.sum(X_for_cos*Y_for_cos, -1)

        elif self.score_func == "bias_product":
            B = torch.index_select(bias_map, 0, bias_ids).view(Y_batch_size, -1)
            return torch.sum(X*Y, -1) + B

        else:
            return torch.sum(X*Y, -1)

    def get_generation_loss(self, example_ids, example_emb_batch, example_word_map, example_lens, cadidate_distribution):
        '''
        :param example_ids: indices of item or user
        :param example_emb_batch: embeddings of item or user
        :param example_word_map: item title list or user word list
        :param example_lens: item tiele lens or user word lens
        :param cadidate_distribution: word distribution
        :return: NCE Loss of (u,w) or (i,w) pair
        '''
        batch_size = example_emb_batch.size()[0]
        
        example_word_index_batch = torch.index_select(example_word_map, 0, example_ids).view(-1).long()
        example_word_batch = torch.index_select(self.word_emb_with_zero, 0, example_word_index_batch).view(batch_size,-1,self.emb_dim)
        example_lens_batch = torch.index_select(example_lens, 0, example_ids)
        example_emb_batch = example_emb_batch.view(batch_size, -1, self.emb_dim)

        sim_score = self.get_sim_score(example_emb_batch, example_word_batch, example_word_index_batch, self.word_bias_with_zero)

        mask_weight = sequence_mask_lt(example_lens_batch,sim_score.size()[1]).float()

        sim_score = sim_score.view(-1)
        loss_func = nn.BCEWithLogitsLoss(weight=mask_weight.view(-1))
        pos_loss = loss_func(sim_score, torch.ones_like(sim_score).float().to(self.device))
        
        noise_ditribution = cadidate_distribution ** self.noise_rate
        
        noise_ids = torch.LongTensor(list(Data.WeightedRandomSampler(noise_ditribution, example_word_index_batch.size()[0]*self.neg_sample_num, True))).to(self.device)
        noise_word_batch = torch.index_select(self.word_embedding.weight, 0 ,noise_ids).view(batch_size, -1, self.emb_dim)

        noise_sim_score = self.get_sim_score(example_emb_batch, noise_word_batch, noise_ids, self.word_bias_with_zero)

        noise_lens_batch = example_lens_batch*self.neg_sample_num
        noise_mask_weight = sequence_mask_lt(noise_lens_batch, noise_sim_score.size()[1]).float().view(batch_size, -1, 1)
        
        noise_sim_score = noise_sim_score.view(-1)
        noise_loss_func = nn.BCEWithLogitsLoss(weight=noise_mask_weight.view(-1))
        neg_loss = noise_loss_func(noise_sim_score, torch.zeros_like(noise_sim_score).float().to(self.device))

        return pos_loss+neg_loss, [example_emb_batch, example_word_batch, noise_word_batch*noise_mask_weight]
    
    def get_uqi_loss(self, personalized_query, item_ids, item_emb_batch, item_distribution):
        '''
        :param personalized_query: projected query + user embedding
        :param item_ids: indices of positive sample item
        :param item_emb_batch: embeddings of positive sample item
        :param item_distribution: item distribution
        :return: NCE Loss of (u,q,i) pair
        '''
        batch_size = item_emb_batch.size()[0]

        sim_score = self.get_sim_score(personalized_query, item_emb_batch, item_ids, self.item_bias_with_zero).view(-1)
        pos_loss = self.BCELoss_fun(sim_score, torch.ones_like(sim_score))
        
        noise_ditribution = item_distribution ** self.noise_rate
        noise_ids = torch.LongTensor(list(Data.WeightedRandomSampler(noise_ditribution, batch_size*self.neg_sample_num, True))).to(self.device)
        noise_item_batch = torch.index_select(self.item_embedding.weight, 0, noise_ids).view(batch_size,-1,self.emb_dim)

        noise_sim_score = self.get_sim_score(personalized_query, noise_item_batch, noise_ids, self.item_bias_with_zero).view(-1)
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
        item_ids = torch.arange(0, self.item_embedding.weight.size()[0]).long().to(self.device)
        sim_score = self.get_sim_score(personalized_query, self.item_embedding.weight, item_ids, self.item_bias).view(-1)

        return sim_score
