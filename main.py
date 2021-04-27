import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import time
import os
from utils.tools import load_list
from utils.tools import load_json
from utils.test import test_all_items

from structures.HEM import HEM
from structures.ZAM import ZAM
from structures.HRNN import HRNN_simple


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing graph embedding for personalized search',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--structure_name', default="HEM")
    parser.add_argument('--score_function', default="product")
    parser.add_argument('--data_root', default="datasets")
    parser.add_argument('--save_root', default="models")
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--att_hidden_units_num', default=5, type=int)
    parser.add_argument('--RNN_layers_num', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--EPOCH', default=20, type=int)
    parser.add_argument('--LR', default=0.01, type=float)
    parser.add_argument('--LAMBDA', default=0.5, type=float)
    parser.add_argument('--L2_weight', default=1e-5, type=float)
    parser.add_argument('--neg_sample_num', default=10, type=int)
    parser.add_argument('--noise_rate', default=0.75, type=float)
    parser.add_argument('--is_val', default=True, type=bool)

    return parser.parse_args(args)


def load_data(structure_name, data_root):
    '''
    :param structure_name: structure name of pps model, including: HEM, ZAM, HRNN
    :param data_root: root file name of dataset
    :return: data dict containing all data needed
    '''

    data_dict = {}

    if structure_name == "HEM" or structure_name == "ZAM":
        data_dir = os.path.join(data_root, "forEM")
    elif structure_name == "HRNN":
        data_dir = os.path.join(data_root, "forHRNN")

    item_title_map_path = os.path.join(data_dir, "item_title_map.txt")
    user_word_map_path = os.path.join(data_dir, "user_word_map.txt")
    user_click_map_path = os.path.join(data_dir, "user_click_map.txt")
    word_distribution_path = os.path.join(data_dir, "word_distribution.txt")
    item_distribution_path = os.path.join(data_dir, "item_distribution.txt")
    click_test_path = os.path.join(data_dir, "click_test.txt")
    click_train_path = os.path.join(data_dir, "click_train.txt")
    click_valid_path = os.path.join(data_dir, "click_valid.txt")
    query_seg_map_path = os.path.join(data_dir, "query_seg.txt")
    query_click_path = os.path.join(data_dir, "query_click.txt")
    query_click_lens_path = os.path.join(data_dir, "query_click_lens.txt")
    query_short_his_path = os.path.join(data_dir, "query_short_his.txt")
    query_short_lens_path = os.path.join(data_dir, "query_short_lens.txt")
    query_long_his_path = os.path.join(data_dir, "query_long_his.txt")
    query_long_lens_path = os.path.join(data_dir, "query_long_lens.txt")
    statistic_path = os.path.join(data_dir, "statistic.json")

    data_dict["statistic_map"] = load_json(statistic_path)
    query_seg_map_with_lens = torch.Tensor(load_list(query_seg_map_path))
    data_dict["query_seg_map"] = query_seg_map_with_lens[:, 1::]
    data_dict["query_seg_lens"] = query_seg_map_with_lens[:, 0]
    data_dict["train_set"] = load_list(click_train_path)
    data_dict["valid_set"] = load_list(click_valid_path)
    data_dict["test_set"] = load_list(click_test_path)

    if structure_name == "HEM" or "ZAM":
        data_dict["word_distribution"] = torch.Tensor(load_list(word_distribution_path)).view(-1)
        data_dict["item_distribution"] = torch.Tensor(load_list(item_distribution_path)).view(-1)
        user_word_map_with_lens = torch.Tensor(load_list(user_word_map_path))
        data_dict["user_word_map"] = user_word_map_with_lens[:, 1::]
        data_dict["user_word_lens"] = user_word_map_with_lens[:, 0]
        item_title_map_with_lens = torch.Tensor(load_list(item_title_map_path))
        data_dict["item_title_map"] = item_title_map_with_lens[:, 1::]
        data_dict["item_title_lens"] = item_title_map_with_lens[:, 0]
        user_click_map_with_lens = torch.Tensor(load_list(user_click_map_path))
        data_dict["user_click_map"] = user_click_map_with_lens[:, 1::]
        data_dict["user_click_lens"] = user_click_map_with_lens[:, 0]

    elif structure_name == "HRNN":
        data_dict["query_click"] = torch.Tensor(load_list(query_click_path))
        data_dict["query_click_lens"] = torch.Tensor(load_list(query_click_lens_path))
        data_dict["query_short_his"] = torch.Tensor(load_list(query_short_his_path))
        data_dict["query_short_lens"] = torch.Tensor(load_list(query_short_lens_path))
        data_dict["query_long_his"] = torch.Tensor(load_list(query_long_his_path))
        data_dict["query_long_lens"] = torch.Tensor(load_list(query_long_lens_path))

    return data_dict


def load_model(args, data_dict):
    emb_dim = args.emb_dim
    neg_sample_num = args.neg_sample_num
    noise_rate = args.noise_rate
    LAMBDA = args.LAMBDA
    L2_weight = args.L2_weight
    score_func = args.score_function
    device = torch.device(args.device)

    word_num = data_dict["statistic_map"]["word_num"]
    item_num = data_dict["statistic_map"]["item_num"]

    model = None
    if args.structure_name == "HEM":
        user_num = data_dict["statistic_map"]["user_num"]

        model = HEM(word_num, item_num, user_num, emb_dim, data_dict["user_word_map"], data_dict["item_title_map"],
                    data_dict["query_seg_map"], data_dict["user_word_lens"], data_dict["item_title_lens"],
                    data_dict["query_seg_lens"], data_dict["word_distribution"], data_dict["item_distribution"],
                    neg_sample_num, noise_rate, LAMBDA, L2_weight, score_func, device).to(device)

    elif args.structure_name == "ZAM":
        user_num = data_dict["statistic_map"]["user_num"]
        behavior_len = data_dict["statistic_map"]["behavior_len"]

        att_hidden_units_num = args.att_hidden_units_num
        model = ZAM(word_num, item_num, user_num, emb_dim, behavior_len, att_hidden_units_num,
                    data_dict["user_word_map"], data_dict["item_title_map"], data_dict["query_seg_map"],
                    data_dict["user_click_map"], data_dict["user_word_lens"], data_dict["item_title_lens"],
                    data_dict["query_seg_lens"], data_dict["user_click_lens"] , data_dict["word_distribution"],
                    data_dict["item_distribution"], neg_sample_num, noise_rate, LAMBDA, L2_weight, score_func,
                    device).to(device)

    elif args.structure_name == "HRNN_simple":
        num_layers = args.RNN_layers_num
        model = HRNN_simple(word_num, item_num, emb_dim, num_layers, data_dict["query_seg_map"],
                           data_dict["query_seg_lens"], data_dict["query_click"], data_dict["query_click_lens"],
                           data_dict["query_long_his"], data_dict["query_short_his"], data_dict["query_long_lens"],
                           data_dict["query_short_lens"], neg_sample_num, device).to(device)

    return model


def run():
    args = parse_args()
    data_dict = load_data(args.structure_name, args.data_root)
    model = load_model(args, data_dict)
    print(model)
    item_num = data_dict["statistic_map"]["item_num"]
    device = torch.device(args.device)
    save_path = os.path.join(args.save_root, args.structure_name)
    optimizer = torch.optim.Adam(model.paramters(), lr=args.LR)
    torch_dataset = Data.TensorDataset(data_dict["train_set"])
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    start = time.time()
    for epoch in range(args.EPOCH):
        for step, [b_x, ] in enumerate(train_loader):
            loss = model(b_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 300 == 0:
                print('Epoch: ', epoch, 'Step: ', step, "loss: %.4f" % loss.cpu().data.numpy())
                end = time.time()
                print("time cost 300 steps:%.2f seconds" % (end - start))
                start = time.time()

        if epoch % args.test_epoch == 0 and args.is_val:
            start2 = time.time()
            model.eval()
            ndcg_at5 = test_all_items(model, data_dict["valid_set"], False, item_num, device)
            if ndcg_at5 > max_ndcg_at5:
                torch.save(model.state_dict(), save_path)
                print("save model at %d epoch" % epoch)
                max_ndcg_at5 = ndcg_at5
            model.train()
            end2 = time.time()
            print("run for valid set:%.2f seconds" % (end2 - start2))

    test_all_items(model, data_dict["test_set"], True, item_num, device)


if "__name__" == "__main__":
    run()