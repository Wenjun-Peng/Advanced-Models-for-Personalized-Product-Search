import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from tools import load_list
from tools import load_json


def test_all_items(model, data_set, is_show, item_num, device):
    '''
    :param model: model for test
    :param data_set: test data: could be test set or valid set
    :param is_show: whether print test info for every user-query pair
    :param item_num: number of item
    :param device: running device
    :return: none
    '''
    test_set = data_set.numpy().to_list()

    ndcg_at5s = []
    ndcg_at10s = []
    Hit_at5s = []
    Hit_at10s = []
    HR_at5s = []
    HR_at10s = []

    # init with tensor for parallel
    decay_ratios = torch.Tensor([np.log2(i + 2) for i in range(10)])
    res = torch.zeros(item_num)
    lastQid = test_set[0][1]
    test_set.append([-1, -1, -1, -1, -1, -1, -1, -1])
    # res_copy = copy.deepcopy(res)
    idcg_at5 = 0
    idcg_at10 = 0
    count = 0
    for info_needed in test_set:
        query_index = info_needed[1]
        user_index = info_needed[0]

        # check if it is end
        if user_index == -1:
            break

        item_index = int(info_needed[2])
        res[item_index] = 1

        # every row means a click
        if count < 5:
            idcg_at5 += 1 / np.log2(count + 2)

        if count < 10:
            idcg_at10 += 1 / np.log2(count + 2)

        count += 1

        # calculate ndcg
        if lastQid != query_index:

            u_id = torch.LongTensor([user_index]).to(device)
            q_id = torch.LongTensor([query_index]).to(device)

            sim_score = model.all_item_test(u_id, q_id)

            # sort items with sim score produced by model
            sortedRes, indices = torch.sort(sim_score.cpu(), 0, descending=True)
            rankRes = torch.index_select(res, 0, indices[0:10]).view(-1)
            # print(u_id,q_id,indices[0:10])
            # calculate ndcg
            single_dcg = rankRes / decay_ratios
            rankRes = rankRes.cpu()
            single_dcg = single_dcg.cpu()
            dcg_at5 = torch.sum(single_dcg[0:5]).item()
            dcg_at10 = torch.sum(single_dcg).item()
            hit_at5 = torch.sum(rankRes[0:5]).item()
            hit_at10 = torch.sum(rankRes).item()

            ndcg_at5 = (dcg_at5 / idcg_at5)
            ndcg_at10 = (dcg_at10 / idcg_at10)
            HR_at5 = (hit_at5 / count)
            HR_at10 = (hit_at10 / count)

            ndcg_at5s.append(ndcg_at5)
            ndcg_at10s.append(ndcg_at10)
            Hit_at5s.append(hit_at5)
            Hit_at10s.append(hit_at10)
            HR_at5s.append(HR_at5)
            HR_at10s.append(HR_at10)

            if is_show:
                print("uid:", user_index, " qid:", query_index, " ndcg@5:%.2f" % ndcg_at5, " ndcg@10:%.2f" % ndcg_at10,
                      " HR@5:%.2f" % HR_at5, " HR@10:%.2f" % HR_at10)

            res = torch.zeros(item_num)
            lastQid = query_index
            idcg_at5 = 0
            idcg_at10 = 0
            count = 0

    print("ndcg@5:%.4f" % np.mean(ndcg_at5s), "ndcg@10:%.4f" % np.mean(ndcg_at10s))
    print("hit@5:%.4f" % np.mean(Hit_at5s), "hit@10:%.4f" % np.mean(Hit_at10s))
    print("HR@5:%.4f" % np.mean(HR_at5s), "HR@10:%.4f" % np.mean(HR_at10s))
    return np.mean(ndcg_at5s)
