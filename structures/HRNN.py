import torch
import torch.nn as nn

from struct_functions import sequence_mask
from struct_functions import sequence_mask_eq


class HRNN_simple(nn.Module):
    def __init__(self, word_num, item_num, emb_dim, num_layers, query_seg, query_seg_lens, query_click,
                 query_click_lens, query_long_his, query_short_his, long_lens, short_lens, neg_num, device):
        super(HRNN_simple, self).__init__()

        self.word_num = word_num
        self.item_num = item_num
        self.emb_dim = emb_dim
        self.num_layers = num_layers

        self.query_click = query_click.to(device)
        self.query_long_his = query_long_his.to(device)
        self.query_short_his = query_short_his.to(device)
        self.query_seg = query_seg.to(device)
        self.query_seg_lens = query_seg_lens.to(device)

        self.long_behavior_len = query_long_his.size()[1]
        self.short_behavior_len = query_short_his.size()[1]
        self.max_seg_len = query_seg.size()[1]
        self.max_click_len = query_click.size()[1]

        self.short_lens = short_lens.to(device)
        self.long_lens = long_lens.to(device)
        self.query_click_lens = query_click_lens.to(device)

        self.neg_num = neg_num

        self.device = device

        self.word_embedding = nn.Embedding(
            num_embeddings=word_num,
            embedding_dim=emb_dim
        )

        self.item_embedding = nn.Embedding(
            num_embeddings=item_num,
            embedding_dim=emb_dim
        )

        self.attention_MLP = nn.Sequential(
            nn.Linear(
                in_features=emb_dim * 2,
                out_features=512,
            ),
            nn.Linear(
                in_features=512,
                out_features=1,

            ),
            nn.Tanh()
        )

        self.short_term_GRU = nn.GRUCell(
            input_size=emb_dim * 2,
            hidden_size=emb_dim,
        )

        self.long_term_GRU = nn.GRUCell(
            input_size=emb_dim * 2,
            hidden_size=emb_dim,
        )

        self.short_interest_project = nn.Linear(
            in_features=emb_dim,
            out_features=emb_dim,
            bias=False,
        )

        self.long_interest_project = nn.Linear(
            in_features=emb_dim,
            out_features=emb_dim,
            bias=False,
        )

        self.query_project = nn.Sequential(
            nn.Linear(
              in_features=emb_dim,
              out_features=emb_dim,
            ),
            nn.Tanh()
        )

        self.loss_func = nn.BCEWithLogitsLoss()

    def append_zero_vec(self):
        zero_vec = torch.zeros(1,self.emb_dim).to(self.device)
        self.item_emb_with_zero = torch.cat([self.item_embedding.weight, zero_vec], 0)
        self.word_emb_with_zero = torch.cat([self.word_embedding.weight, zero_vec], 0)

    def get_short_interest(self, batch_size, query_ids):
        # [batch_size * behavior_len]
        short_ids = torch.index_select(self.query_short_his, 0, query_ids).view(batch_size, -1).view(-1).long()

        # [batch_size * behavior_len * max_seg_len]
        query_short_word_ids = torch.index_select(self.query_seg, 0, short_ids).view(-1).long()
        query_short_word_batch = torch.index_select(self.word_emb_with_zero, 0, query_short_word_ids).view(batch_size, self.short_behavior_len,
                                                                                self.max_seg_len, self.emb_dim)

        # [batch_size * behavior_len]
        
        short_lens_batch = torch.index_select(self.query_seg_lens, 0, short_ids).view(-1)
        short_word_mask = sequence_mask(short_lens_batch, self.max_seg_len).view(batch_size,
                                                                                     self.short_behavior_len,
                                                                                     self.max_seg_len)

        short_word_weight = torch.ones(batch_size, self.short_behavior_len, self.max_seg_len).to(self.device)

        masked_short_word_weight = short_word_weight.masked_fill(short_word_mask, 0).view(batch_size,
                                                                                          self.short_behavior_len,
                                                                                          self.max_seg_len, 1)
        query_short_emb_batch = self.query_project(torch.sum(masked_short_word_weight * query_short_word_batch, 2) / short_lens_batch.view(
            batch_size, self.short_behavior_len, 1))

        # [batch_size * behavior_len * max_click_len]
        short_click_ids = torch.index_select(self.query_click, 0, short_ids).view(-1).long()
        agg_short_item_batch = torch.index_select(self.item_emb_with_zero, 0, short_click_ids).view(batch_size, self.short_behavior_len,
                                                                                  self.max_click_len, self.emb_dim)

        short_item_weight = torch.ones(batch_size, self.short_behavior_len, self.max_click_len).to(self.device)
        short_item_lens = torch.index_select(self.query_click_lens, 0, short_ids).view(-1)
        short_item_mask = sequence_mask(short_item_lens, self.max_click_len).view(batch_size,
                                                                                      self.short_behavior_len,
                                                                                      self.max_click_len)
        masked_short_item_weight = short_item_weight.masked_fill(short_item_mask, 0).view(batch_size,
                                                                                          self.short_behavior_len,
                                                                                          self.max_click_len, 1)

        # [batch_size * behavior_len * emb_dim]
        short_item_batch = torch.sum(agg_short_item_batch * masked_short_item_weight, 2) / short_item_lens.view(
           batch_size, self.short_behavior_len, 1)

        # [batch_size * behavior_len * 2*emb_dim]
        short_input = torch.cat([query_short_emb_batch, short_item_batch], 2)

        h_state = torch.zeros(batch_size, self.emb_dim).to(self.device)
        short_states_list = [ ]
        for time_step in range(self.short_behavior_len):
            h_state = self.short_term_GRU(short_input[:, time_step, :], h_state)
            short_states_list.append(h_state.view(batch_size, 1, -1))

        # [batch_size * behavior_len * emb_dim]
        short_states = torch.cat(short_states_list, 1)

        # [batch_size]
        short_interest_ids = torch.index_select(self.short_lens-1, 0, query_ids).view(-1).long()
        interest_mask = sequence_mask_eq(short_interest_ids, self.short_behavior_len).view(batch_size, -1, 1)

        short_interest = torch.masked_select(short_states, interest_mask).view(batch_size, self.emb_dim)
        projected_short_interest = self.short_interest_project(short_interest)
        return projected_short_interest

    def get_long_interest(self, batch_size, query_ids, query_emb_batch):

        # construct query_emb in long session
        # [batch_size * behavior_len]
        long_ids = torch.index_select(self.query_long_his, 0, query_ids).view(batch_size, -1).view(-1).long()

        # [batch_size * behavior_len * query_word_Len]
        query_long_word_ids = torch.index_select(self.query_seg, 0, long_ids).view(-1).long()

        # [batch_size * behavior_len * query_word_Len * emb_dim]
        query_long_word_batch = torch.index_select(self.word_emb_with_zero, 0, query_long_word_ids).view(batch_size, self.long_behavior_len,
                                                                              self.max_seg_len,
                                                                              self.emb_dim)

        long_word_weight = torch.ones(batch_size, self.long_behavior_len, self.max_seg_len).to(self.device)

        # [batch_size * behavior_len * query_word_Len]
        long_lens_batch = torch.index_select(self.query_seg_lens, 0, long_ids).view(-1)
        long_word_mask = sequence_mask(long_lens_batch, self.max_seg_len).view(batch_size, self.long_behavior_len,
                                                                                   self.max_seg_len)
        masked_long_word_weight = long_word_weight.masked_fill(long_word_mask, 0).view(batch_size,
                                                                                       self.long_behavior_len,
                                                                                       self.max_seg_len, 1)

        # [batch_size * behavior_len * emb_dim]
        query_long_emb_batch = self.query_project(torch.sum(masked_long_word_weight * query_long_word_batch, 2) / long_lens_batch.view(
            batch_size, self.long_behavior_len, 1))

        # construct item_emb in long session
        # [batch_size * behavior_len * max_click_len]
        long_click_ids = torch.index_select(self.query_click, 0, long_ids).view(-1).long()

        # [batch_size * behavior_len * max_click_len * emb_dim]
        agg_long_item_batch = torch.index_select(self.item_emb_with_zero, 0, long_click_ids).view(batch_size, self.long_behavior_len, -1,
                                                                                self.emb_dim)

        long_item_weight = torch.ones(batch_size, self.long_behavior_len, self.max_click_len).to(self.device)

        long_item_lens = torch.index_select(self.query_click_lens, 0, long_ids).view(-1)
        long_item_mask = sequence_mask(long_item_lens, self.max_click_len).view(batch_size, self.long_behavior_len,
                                                                                    self.max_click_len)

        masked_long_item_weight = long_item_weight.masked_fill(long_item_mask, 0).view(batch_size,
                                                                                       self.long_behavior_len,
                                                                                       self.max_click_len, 1)

        # [batch_size * behavior_len * emb_dim]
        long_item_batch = torch.sum(agg_long_item_batch * masked_long_item_weight, 2) / long_item_lens.view(batch_size,
                                                                                                             self.long_behavior_len,
                                                                                                             1)
        # [batch_size * behavior_len * 2*emb_dim]
        long_input = torch.cat([query_long_emb_batch, long_item_batch], 2)

        h_state = torch.zeros(batch_size, self.emb_dim).to(self.device)
        long_states_list = []
        for time_step in range(self.long_behavior_len):
            h_state = self.long_term_GRU(long_input[:, time_step, :], h_state)
            long_states_list.append(h_state.view(batch_size, 1, -1))

        # [batch_size * behavior_len * emb_dim]
        long_states = torch.cat(long_states_list, 1).view(batch_size, self.long_behavior_len, -1)
        # [batch_size * behavior_len * self.emb_dim]
        query_emb_batch_for_long = query_emb_batch.view(batch_size,1,self.emb_dim).repeat(1,self.long_behavior_len,1)

        # [batch_size * behavior_len * 2*self.emb_dim]
        query_and_state = torch.cat([query_emb_batch_for_long, long_states], 2).view(
            batch_size * self.long_behavior_len, 2 * self.emb_dim)

        # [batch_size * behavior_len]
        e = self.attention_MLP(query_and_state).view(batch_size, self.long_behavior_len)
        # [batch_size]

        long_lens_batch = torch.index_select(self.long_lens, 0, query_ids).view(-1)
        e_mask = sequence_mask(long_lens_batch, self.long_behavior_len)

        masked_a = e.masked_fill(e_mask, -1e9)
        a = torch.softmax(masked_a, 1).view(batch_size, self.long_behavior_len, 1)
        long_interest = torch.sum(long_states * a, 1)
        projected_long_interest = self.long_interest_project(long_interest)
        return projected_long_interest

    def get_neg_sample_loss(self, batch_size, query_emb_batch, short_interest, long_interest, item_emb_batch):
        short_term_score = self.get_sim_score(short_interest, item_emb_batch)
        long_term_score = self.get_sim_score(long_interest, item_emb_batch)
        query_item_score = self.get_sim_score(query_emb_batch, item_emb_batch)
        positive_score = 0.7*short_term_score + 0.3*long_term_score + query_item_score
        positive_label = torch.ones_like(positive_score).to(self.device)
        positive_loss = self.loss_func(positive_score, positive_label)

        negative_item_ids = torch.randint(0, self.item_num, (batch_size*self.neg_num,)).to(self.device)
        negative_item_batch = self.item_embedding(negative_item_ids)
        negative_short_term_score = self.get_sim_score(short_interest.view(batch_size,1,self.emb_dim).repeat(1, self.neg_num, 1).view(-1, self.emb_dim), negative_item_batch)
        negative_long_term_score = self.get_sim_score(long_interest.view(batch_size,1,self.emb_dim).repeat(1, self.neg_num, 1).view(-1, self.emb_dim), negative_item_batch)
        negative_query_item_score = self.get_sim_score(query_emb_batch.view(batch_size,1,self.emb_dim).repeat(1, self.neg_num, 1).view(-1, self.emb_dim), negative_item_batch)
        negative_score = 0.7*negative_short_term_score + 0.3*negative_long_term_score + negative_query_item_score
        negative_label = torch.zeros_like(negative_score).to(self.device)
        negative_loss = self.loss_func(negative_score, negative_label)

        return positive_loss+negative_loss


    def get_sim_score(self, interest, item):
        X = interest / torch.sqrt(torch.sum(interest * interest, 1)).view(-1, 1)
        Y = item / torch.sqrt(torch.sum(item * item, 1)).view(-1, 1)
        # X = interest
        # Y = item
        return torch.sum(X * Y, 1)

    def forward(self, x):
        self.append_zero_vec()
        batch_size = x.size()[0]

        query_ids = x[:, 1].long()
        item_ids = x[:, 2].long()
        item_emb_batch = torch.index_select(self.item_emb_with_zero, 0, item_ids)
        query_word_weight = torch.ones(batch_size, self.max_seg_len).to(self.device)
        query_word_ids = torch.index_select(self.query_seg, 0, query_ids).view(-1).long()
        query_word_batch = torch.index_select(self.word_emb_with_zero, 0, query_word_ids).view(batch_size, -1, self.emb_dim)
        query_lens_batch = torch.index_select(self.query_seg_lens, 0, query_ids)

        # [batch_size * max_seg_len]
        query_word_mask = sequence_mask(query_lens_batch, self.max_seg_len)
        masked_query_word_weight = query_word_weight.masked_fill(query_word_mask, 0).view(batch_size, self.max_seg_len,
                                                                                          1)

        query_emb_batch = self.query_project(torch.sum(masked_query_word_weight * query_word_batch, 1) / query_lens_batch.view(batch_size,
                                                                                                            1))
        short_interest = self.get_short_interest(batch_size, query_ids)
        long_interest = self.get_long_interest(batch_size, query_ids, query_emb_batch)

        neg_sample_loss = self.get_neg_sample_loss(batch_size, query_emb_batch, short_interest, long_interest, item_emb_batch)

        return neg_sample_loss

    def all_item_test(self, u_id, q_id):
        self.append_zero_vec()
        batch_size = 1
        query_word_ids = torch.index_select(self.query_seg, 0, q_id).view(-1).long()
        query_word_batch = torch.index_select(self.word_emb_with_zero, 0, query_word_ids).view(batch_size, -1, self.emb_dim)
        query_word_weight = torch.ones(batch_size, self.max_seg_len).to(self.device)
        query_lens_batch = torch.index_select(self.query_seg_lens, 0, q_id)
        query_word_mask = sequence_mask(query_lens_batch, self.max_seg_len).to(self.device)
        masked_query_word_weight = query_word_weight.masked_fill(query_word_mask, 0).view(batch_size, self.max_seg_len,
                                                                                          1)
        query_emb_batch = self.query_project(torch.sum(masked_query_word_weight * query_word_batch, 1) / query_lens_batch.view(batch_size, 1))
        short_interest = self.get_short_interest(batch_size, q_id)
        long_interest = self.get_long_interest(batch_size, q_id, query_emb_batch)

        short_term_score = self.get_sim_score(short_interest, self.item_embedding.weight)
        long_term_score = self.get_sim_score(long_interest, self.item_embedding.weight)
        query_item_score = self.get_sim_score(query_emb_batch, self.item_embedding.weight)
        
        return 0.7*short_term_score + 0.3*long_term_score + query_item_score
