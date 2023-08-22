import math
import pickle
import sys
from collections import defaultdict
import torch
import numpy as np
import random
from copy import deepcopy
from scipy import sparse
import faiss

def write_pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_seq_user(train_file):
    res_item = defaultdict(list)
    res_sliceid = defaultdict(list)
    with open(train_file, 'r', encoding='utf8') as f:
        next(f)
        for line in f:
            hsty_per = []
            slice_per = []
            line = line.strip()
            user_id = int(line.split(':')[0])
            item_seq = line.split(':')[1].split('|')
            for sliceid, items in enumerate(item_seq):
                if items == '':
                    continue
                items = items.split(',')
                items = [int(item) for item in items]
                hsty_per.extend(items)
                slice_per.extend([sliceid]*len(items))
            res_item[user_id] = hsty_per
            res_sliceid[user_id] = slice_per

    return res_item, res_sliceid




def get_user_history_eval(val_file, test_file):
    user_item_dict = defaultdict(list)
    with open(val_file, 'r', encoding='utf8') as f:
        next(f)
        for line in f:
            line = line.strip()
            user_id = int(line.split(':')[0])
            item_seq = line.split(':')[1].split('|')
            u_history = []
            for items in item_seq:
                if items == '':
                    continue
                items = items.split(',')
                for item in items:
                    u_history.append(int(item))
            user_item_dict[user_id] = u_history

    with open(test_file, 'r', encoding='utf8') as f:
        next(f)
        for line in f:
            line = line.strip()
            user_id = int(line.split(':')[0])
            item_seq = line.split(':')[1].split('|')
            u_history = []
            for items in item_seq:
                if items == '':
                    continue
                items = items.split(',')
                for item in items:
                    u_history.append(int(item))
            user_item_dict[user_id] = u_history

    return user_item_dict
def relative_slice_item(num_items, train_file):
    res_list = [[] for _ in range(num_items+1)]
    with open(train_file, 'r', encoding='utf8') as f:
        next(f)
        for line in f:
            line = line.strip()
            user_id = int(line.split(':')[0])
            item_seqs = line.split(':')[1].split('|')
            for i, items in enumerate(item_seqs):
                if items == '':
                    continue
                items = items.split(',')
                for item in items:
                    if i not in res_list[int(item)]:
                        res_list[int(item)].append(i)
    for i in range(len(res_list)):
        res_list[i] = sorted(res_list[i])
    return np.array(res_list, dtype=object)

def get_rela_item_list(num_slices, train_file):
    res_list = [[] for _ in range(num_slices-1)]
    with open(train_file, 'r', encoding='utf8') as fr:
        next(fr)
        for line in fr:
            line = line.strip()
            user_id = int(line.split(':')[0])
            item_seqs = line.split(':')[1].split('|')
            for i,items in enumerate(item_seqs):
                if items == '':
                    continue
                items = items.split(',')
                for item in items:
                    if int(item) not in res_list[i]:
                        res_list[i].append(int(item))
        for i in range(len(res_list)):
            res_list[i] = sorted(res_list[i])
    return res_list

def get_irrelative_item_id_list(num_items, all_list):
    res_list = []
    all_items = np.arange(1, num_items+1)
    for item_list in all_list:

        sub_res_list = list(set(all_items)-set(item_list))
        sub_res_list = [i-1 for i in sub_res_list]
        res_list.append(sub_res_list)

    return res_list

def construct_init_iig_sql(num_slices, num_items, train_file):

    A_list = []
    edge_list_all = [[] for _ in range(num_slices-1)]
    with open(train_file, 'r', encoding='utf8') as f:
        next(f)
        for i, line in enumerate(f):
            line = line.strip()
            item_seq = line.split(':')[1].split('|')
            for slice_id, items in enumerate(item_seq):
                if items == '':
                    continue
                items = items.split(',')
                items = [int(item) for item in items]
                edge_list_all[slice_id].extend([(items[i], items[i+1]) for i in range(len(items)-1)])

    for i,edge_list in enumerate(edge_list_all):
        edge_list_all[i] = list(set(edge_list))

    for i in range(num_slices-1):
        row = np.array([l[0]-1 for l in edge_list_all[i]])
        column = np.array([l[1]-1 for l in edge_list_all[i]])
        data = np.ones(len(edge_list_all[i]))
        A = sparse.coo_matrix((data, (row, column)), shape=(num_items, num_items))
        A_list.append(A)

    return A_list

def slice_to_onehot(num_slices, slice_array):
    res_tensor = torch.zeros((slice_array.shape[0], num_slices-1))
    for user_id, l in enumerate(slice_array):
        for time_id in l:
            res_tensor[user_id][time_id] = 1

    return res_tensor


def zero_to_one(sum_tensor):
    res = deepcopy(sum_tensor)
    for i,j in enumerate(sum_tensor):
        if j == 0:
            res[i] = 1
    return res

def clip_slice_int(slice_onehot, id_int):
    res = deepcopy(slice_onehot)
    res[:,id_int+1:] = 0
    return res

def compute_average_rep(all_emb, item_slices_train, num_slices):
    res = []
    slice_item_onehot = slice_to_onehot(num_slices=num_slices, slice_array=item_slices_train)
    for t in range(num_slices-1):
        slice_item_onehot_clip = clip_slice_int(slice_onehot=slice_item_onehot, id_int=t)
        slice_item_num = zero_to_one(slice_item_onehot_clip.sum(1)).unsqueeze(1)
        res_slice = (slice_item_onehot_clip.t().unsqueeze(2) * all_emb).sum(0)/slice_item_num
        res.append(res_slice)
    res = torch.stack(res, dim=0)

    return res


def normalize_L2(x):
    faiss.fvec_renorm_L2(x.shape[1], x.shape[0], faiss.swig_ptr(x))

def find_closest_nodes(Emb, k, emb_dim):
    E = deepcopy(Emb)
    normalize_L2(E)
    d = emb_dim
    n_list = 20
    res = faiss.StandardGpuResources()
    quantizer = faiss.IndexFlatIP(d)
    index_cpu = faiss.IndexIVFFlat(quantizer, d, n_list)
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index_gpu.train(E[1:])
    index_gpu.add(E[1:])
    index_gpu.nprobe = 10
    _, I = index_gpu.search(E[1:], k+1)
    row = np.repeat(np.array([i for i in range(I.shape[0])]), k)
    column = I[:,1:].flatten()
    data = np.ones((I.shape[0] * k))
    res = sparse.coo_matrix((data, (row,column)), shape=(I.shape[0], I.shape[0]))
    return res


def get_trend_rep(comms, max_comm, emb_dim, E):
    res = np.zeros((max_comm, emb_dim))
    for i,com in enumerate(comms[:max_comm]):
        res[i] = E[[item+1 for item in list(com)]].mean(0)
    return res

def sampler_train(num_items, adj_user, b_users):
    negs = []
    for user in b_users:
        user = int(user[0])
        while True:
            neg_oneuser = random.sample(range(1, num_items+1), 1)[0]
            if neg_oneuser not in adj_user[user]:
                negs.append(neg_oneuser)
                break
    negs = torch.LongTensor(negs)
    return negs

def sampler_eval(num_items, adj_user, b_users, size):
    negs = []
    for user in b_users:
        user = int(user[0])
        houxuan = list(set(np.arange(1, num_items+1))-set(adj_user[user]))
        neg_oneuser = random.sample(houxuan, size)
        negs.append(np.array(neg_oneuser))
    negs = torch.LongTensor(np.array(negs).squeeze())

    return negs

def eval_val(model, trends_item, eval_loader, user_history, comm_size_eval, args):
    valid_choices = 0.0
    ht = 0.0
    ndcg = 0.0

    for i, (pad_seq_batch, pad_seq_sliceid_batch) in enumerate(eval_loader):
        user_batch = pad_seq_batch[-1].unsqueeze(-1)
        tgt_batch = pad_seq_batch[-2].unsqueeze(1)
        seq_batch = pad_seq_batch[:-2]
        seq_batch = torch.stack(seq_batch, dim=0).t()
        neg_batch = sampler_eval(args.num_items, user_history, user_batch, 100)  # (bs, 100)
        cddt_batch = torch.hstack((tgt_batch, neg_batch))
        seq_slice_batch = pad_seq_sliceid_batch
        seq_slice_batch = torch.stack(seq_slice_batch, dim=0).t()
        prediction = -model.predict(trends_item, seq_batch, cddt_batch, seq_slice_batch, comm_size_eval)
        rank = prediction.argsort().argsort()[:, 0].to('cpu')
        valid_choices += user_batch.size(0)
        ht += sum(rank < 10)
        rank_ndcg = 1 / np.log2(rank + 2)
        ndcg += sum(rank_ndcg[rank < 10])

    return ndcg / valid_choices, ht / valid_choices


def eval_example(model, trends_sum, trends_item, pad_seq_batch, pad_seq_sliceid_batch, user_history, args):

    pad_seq_batch = torch.LongTensor(pad_seq_batch)
    pad_seq_sliceid_batch = torch.LongTensor(pad_seq_sliceid_batch)
    user_batch = pad_seq_batch[:,-1].unsqueeze(-1)
    tgt_batch = pad_seq_batch[:,-2].unsqueeze(-1)
    seq_batch = pad_seq_batch[:,:-2]
    neg_batch = sampler_eval(args.num_items, user_history, user_batch, 100)
    cddt_batch = torch.hstack((tgt_batch, neg_batch)).to('cuda')
    seq_slice_batch = pad_seq_sliceid_batch
    prediction = model.predict(trends_sum, trends_item, seq_batch, cddt_batch, seq_slice_batch)
    _, index = torch.topk(prediction, 10, dim=1)
    topk_list = torch.gather(cddt_batch, 1, index)

    rank = (-prediction).argsort().argsort()[:, 0].to('cpu')

    return topk_list, rank

def eval_example_model2(model, pad_seq_batch, user_history, args):
    pad_seq_batch = torch.LongTensor(pad_seq_batch)
    user_batch = pad_seq_batch[:,-1].unsqueeze(-1)
    tgt_batch = pad_seq_batch[:,-2].unsqueeze(-1)
    seq_batch = pad_seq_batch[:,:-2]
    neg_batch = sampler_eval(args.num_items, user_history, user_batch, 100)
    cddt_batch = torch.hstack((tgt_batch, neg_batch)).to('cuda')
    prediction = model.predict(user_batch, seq_batch, cddt_batch)
    _, index = torch.topk(prediction, 10, dim=1)
    topk_list = torch.gather(cddt_batch, 1, index)

    rank = (-prediction).argsort().argsort()[:, 0].to('cpu')

    return topk_list, rank
def compute_optimal_transport(M, r, c, lam, epsilon=1e-5):
    n, m = M.shape
    P = np.exp(- lam * M)
    P /= P.sum()
    u = np.zeros(n)
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / (u+(1e-10))).reshape((-1, 1))
        P *= (c / (P.sum(0)+(1e-10))).reshape((1, -1))
    return P, np.sum(P * M)

