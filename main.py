import os
import sys
import torch
import torch.nn as nn
import numpy as np
from time import time
from utils import *
from model3 import *
import logging
logging.getLogger().setLevel(logging.INFO)
from torch.utils.data import DataLoader
from loader import *
import networkx as nx
import networkx.algorithms.community as netcomm



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=2023, type=str)
parser.add_argument('--dataset', required=True)
parser.add_argument('--epoch', default=5000, type=int)
parser.add_argument('--bs', default=1024, type=int)
parser.add_argument('--bs_eval', default=256, type=int)
parser.add_argument('--emb_size', default=32, type=int)
parser.add_argument('--seq_len', default=20, type=int)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--l2', default=1e-4, type=float)
parser.add_argument('--ot_lam', default=1, type=float)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--tau', default=3, type=float)
parser.add_argument('--scale', default=3, type=float)
parser.add_argument('--k', default=1, type=int)
parser.add_argument('--max_comm', default=30, type=int)
parser.add_argument('--resolution', default=0.9, type=float)
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--mark', default='default', type=str)
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
random_seed(args.seed)


if __name__ == '__main__':
    if not os.path.exists('./data/' + args.dataset + '/results'):
        os.makedirs('./data/' + args.dataset + '/results')
    logging.basicConfig(filename="./data/{}/results/{}.log".format(args.dataset, args.mark))
    logging.info(args)
    if args.dataset == 'Sports_6':
        args.num_users = 17881
        args.num_items = 13589
        args.num_slices = 6
        train_dataset = Sports(mode='train', args=args)
        val_dataset = Sports(mode='val', args=args)
        test_dataset = Sports(mode='test', args=args)
    if args.dataset == 'Phones_6':
        args.num_users = 18958
        args.num_items = 10370
        args.num_slices = 6
        train_dataset = Phones(mode='train', args=args)
        val_dataset = Phones(mode='val', args=args)
        test_dataset = Phones(mode='test', args=args)
    if args.dataset == 'RR_4':
        args.num_users = 5013
        args.num_items = 7735
        args.num_slices = 4
        train_dataset = RR(mode='train', args=args)
        val_dataset = RR(mode='val', args=args)
        test_dataset = RR(mode='test', args=args)
    if args.dataset == 'Yelp_6':
        train_dataset = Yelp(mode='train', args=args)
        val_dataset = Yelp(mode='val', args=args)
        test_dataset = Yelp(mode='test', args=args)
        args.num_users = 18270
        args.num_items = 15467
        args.num_slices = 6

    train_file = './data/' + args.dataset + '/train_seq.txt'
    user_item_dict = get_user_history_eval('./data/' + args.dataset + '/out_val.txt',
                                           './data/' + args.dataset + '/out_test.txt')
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=1, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.bs_eval, shuffle=True, num_workers=1, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs_eval, shuffle=True, num_workers=1, drop_last=False)
    item_slices = relative_slice_item(args.num_items, train_file)
    item_slice_id_list = get_rela_item_list(args.num_slices,
                                            train_file)
    irrelative_item_slice_id_list = get_irrelative_item_id_list(args.num_items, item_slice_id_list)
    iig_list = construct_init_iig_sql(args.num_slices, args.num_items, train_file)
    relu = torch.nn.ReLU()

    model = pureSASrec(args, item_slices)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    num_inter = len(train_loader) * args.bs

    for e in range(args.epoch):
        torch.cuda.empty_cache()
        time_start = time()
        epoch_loss = 0.0
        model.train()

        all_emb = []
        for i in range(args.num_slices - 1):
            item_emb = model.SASs[i].item_embedding.weight
            all_emb.append(item_emb)
        all_emb = torch.stack(all_emb, dim=0).to('cpu')  # (num_slices-1, num_items+1, 32)
        item_average_rep = compute_average_rep(all_emb, item_slices, args.num_slices)
        item_average_rep = item_average_rep.detach().numpy()
        trend_item_embs = []
        trends_num = []

        if e % 10 == 0:
            all_comms = []
            comm_size = []
            for i in range(args.num_slices-1):
                A = iig_list[i]
                item_emb = model.SASs[i].item_embedding.weight
                item_emb = item_emb.cpu().detach().numpy()
                S = find_closest_nodes(item_emb, args.k, args.emb_size)
                G = nx.from_scipy_sparse_array(A + S)
                G.remove_nodes_from(irrelative_item_slice_id_list[i])
                comms = netcomm.louvain_communities(G, resolution=args.resolution)
                G.clear()
                comms = [com for com in comms if len(com) > 20]
                comms = sorted(comms, key=lambda x: len(x), reverse=True)
                comm_size_per_slice = [len(c) for c in comms[:args.max_comm]]
                if len(comms) < args.max_comm:
                    comm_size_per_slice.extend([0 for _ in range(args.max_comm - len(comms))])
                comm_size_per_slice = [item / sum(comm_size_per_slice) for item in comm_size_per_slice]
                comm_size.append(comm_size_per_slice)

                all_comms.append(comms)
            comm_size = torch.FloatTensor(comm_size).to(args.device)
        for i, comms in enumerate(all_comms):
            trends_num.append(min(len(comms), args.max_comm))
            trend_rep = get_trend_rep(comms, args.max_comm, args.emb_size, item_average_rep[i])
            trend_item_embs.append(trend_rep)
        trends_num = torch.from_numpy(np.array(trends_num)).unsqueeze(1).to(args.device)
        trend_item_embs = torch.from_numpy(np.stack(trend_item_embs, axis=0)).to(args.device)
        trend_item_embs = model.emb_dropout_item(trend_item_embs)

        for i, (pad_seq_batch, pad_seq_sliceid_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            user_batch = pad_seq_batch[-1].unsqueeze(-1)  # (bs, 1)
            tgt_batch = pad_seq_batch[-2].unsqueeze(1)  # (bs, 1)
            seq_batch = pad_seq_batch[:-2]
            seq_batch = torch.stack(seq_batch, dim=0).t()  # (bs, seq_len)

            neg_batch = sampler_train(args.num_items, user_item_dict, user_batch)
            neg_batch = neg_batch.unsqueeze(-1)

            seq_slice_batch = pad_seq_sliceid_batch[:-1]
            seq_slice_batch = torch.stack(seq_slice_batch,dim=0).t()  # (bs, seq)
            tgt_slice_batch = pad_seq_sliceid_batch[-1].unsqueeze(-1)  # (bs, 1)
            loss = model(trends_num, trend_item_embs, seq_batch, tgt_batch, neg_batch, seq_slice_batch, tgt_slice_batch, comm_size)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        epoch_loss /= num_inter
        time_end = time()
        print('Epoch {:3d}, loss: {:.8f}, training time: {:.3f}'.format(e, epoch_loss, time_end - time_start))
        if e % 10 == 0:
            model.eval()
            trend_item_embs_eval = []
            trends_num_eval = []

            all_emb = []

            for i in range(args.num_slices - 1):
                item_emb = model.SASs[i].item_embedding.weight
                all_emb.append(item_emb)
            all_emb = torch.stack(all_emb, dim=0).to('cpu')

            item_average_rep = compute_average_rep(all_emb, item_slices, args.num_slices)
            item_average_rep = item_average_rep.detach().numpy()
            comm_size_eval = []
            for i in range(args.num_slices - 1):
                A = iig_list[i]
                item_emb = model.SASs[i].item_embedding.weight
                item_emb = item_emb.cpu().detach().numpy()
                S = find_closest_nodes(item_emb, args.k, args.emb_size)
                G = nx.from_scipy_sparse_array(A + S)
                G.remove_nodes_from(irrelative_item_slice_id_list[i])
                comms = netcomm.louvain_communities(G, resolution=args.resolution)
                G.clear()
                comms = [com for com in comms if len(com) > 20]
                comms = sorted(comms, key=lambda x: len(x), reverse=True)
                comm_size_eval_per_slice = [len(c) for c in comms[:args.max_comm]]
                if len(comms) < args.max_comm:
                    comm_size_eval_per_slice.extend([0 for _ in range(args.max_comm - len(comms))])
                comm_size_eval_per_slice = [item / sum(comm_size_eval_per_slice) for item in
                                            comm_size_eval_per_slice]
                comm_size_eval.append(comm_size_eval_per_slice)

                trends_num_eval.append(min(len(comms), args.max_comm))
                trend_rep = get_trend_rep(comms, args.max_comm, args.emb_size, item_average_rep[i])
                trend_item_embs_eval.append(trend_rep)
            trend_item_embs_eval = torch.from_numpy(np.stack(trend_item_embs_eval, axis=0)).to(args.device)

            logging.info('Epoch: {}'.format(e))
            ndcg_10, hr_10 = eval_val(model, trend_item_embs_eval, val_loader, user_item_dict, comm_size_eval, args)
            print('Val NDCG@10: {:.4f}, HR@10: {:.4f}'.format(ndcg_10, hr_10))
            logging.info('Val NDCG@10: {:.4f}, HR@10: {:.4f}'.format(ndcg_10, hr_10))
            ndcg_10, hr_10 = eval_val(model, trend_item_embs_eval, test_loader, user_item_dict, comm_size_eval, args)
            print('Test NDCG@10: {:.4f}, HR@10: {:.4f}'.format(ndcg_10, hr_10))
            logging.info('Test NDCG@10: {:.4f}, HR@10: {:.4f}'.format(ndcg_10, hr_10))
