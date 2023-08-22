import sys

import torch
import torch.nn as nn
import numpy as np
import pickle
from utils import *
from sklearn.metrics.pairwise import pairwise_distances
import torch.nn.functional as F


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, emb_size, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(emb_size, emb_size, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(emb_size, emb_size, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs



class SASRec_ONE(nn.Module):
    def __init__(self, args):
        super(SASRec_ONE, self).__init__()
        self.num_users = args.num_users
        self.num_items = args.num_items
        self.emb_size = args.emb_size
        self.user_embedding = nn.Embedding(self.num_users + 1, self.emb_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.num_items + 1, self.emb_size, padding_idx=0)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        self.sigmoid = nn.Sigmoid()

class pureSASrec(nn.Module):
    def __init__(self, args, item_slices):
        super(pureSASrec, self).__init__()
        self.device = args.device
        self.args = args
        self.emb_size = args.emb_size
        self.num_slices = args.num_slices
        self.SASs = nn.ModuleList()
        for i in range(self.num_slices-1):
            new_model = SASRec_ONE(args)
            self.SASs.append(new_model)

        self.slice_items = slice_to_onehot(self.num_slices, item_slices).to(self.device)
        self.slice_num_items = torch.from_numpy(
            np.array([len(sublist) if len(sublist) > 0 else 1 for sublist in item_slices])).unsqueeze(1).to(self.device)
        self.emb_dropout_user = nn.Dropout(p=args.dropout)
        self.emb_dropout_item = nn.Dropout(p=args.dropout)
        self.sigmoid = nn.Sigmoid()
        self.position_emb = torch.nn.Embedding(args.seq_len, self.emb_size)
        torch.nn.init.xavier_normal_(self.position_emb.weight)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout)
        self.dropout = args.dropout
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)
        self.num_blocks = 1
        self.num_heads = 1
        self.elu = torch.nn.ELU()
        self.tau = args.tau
        self.ot_lam = args.ot_lam
        self.alpha = args.alpha
        self.scale = args.scale
        self.sftmx = torch.nn.Softmax(dim=1)
        self.sftmx_eval = torch.nn.Softmax(dim=2)
        self.W1 = nn.Linear(self.emb_size, self.emb_size)
        self.W2 = nn.Linear(self.emb_size, self.emb_size)
        nn.init.xavier_normal_(self.W1.weight.data)
        nn.init.xavier_normal_(self.W2.weight.data)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.emb_size, self.num_heads, self.dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.dropout)
            self.forward_layers.append(new_fwd_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

    def log2feats(self, log_seqs, seqs):
        seqs *= self.emb_size ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])  # (bs, seq)
        seqs += self.position_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)  # (bs, seq)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)  # (seq, bs, emb)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def log2feats_eval(self, log_seqs, seqs):
        seqs *= self.emb_size ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])  # (bs, seq)
        seqs += self.position_emb(torch.LongTensor(positions).to(self.device))

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)  # (bs, seq)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats



    def forward(self, trends_num, trends_item, log_seqs, targets, negatives, log_slices, tgt_slices, comm_size):
        item_all_embs = []
        for i in range(self.num_slices - 1):
            item_emb = self.SASs[i].item_embedding.weight
            item_all_embs.append(item_emb)
        item_all_embs = torch.stack(item_all_embs, dim=1)

        log_slice_onehot = self.slice_items[log_seqs]
        flag_onehot = torch.nn.functional.one_hot(log_slices,
                                                  num_classes=self.num_slices - 1)
        col_indices = torch.arange(flag_onehot.size(2)).unsqueeze(0).unsqueeze(0)
        clip = (col_indices <= log_slices.unsqueeze(-1)).long().to(
            self.device)
        res = clip * log_slice_onehot

        seq_all_emb = item_all_embs[log_seqs]
        res = res.unsqueeze(-1)
        masked_m = seq_all_emb * res
        sum_masked_m = torch.sum(masked_m, dim=2)
        sq = torch.sum(res, dim=2)
        sq = sq.masked_fill(sq == 0, 1)
        seq_rep = sum_masked_m / sq

        targets_slice_onehot = self.slice_items[targets.squeeze()]
        col_indices = torch.arange(self.num_slices - 1).unsqueeze(0)
        clip = (col_indices <= tgt_slices).long().to(self.device)
        res_tgt = clip * targets_slice_onehot
        targets_all_emb = item_all_embs[targets.squeeze()]
        tgt_rep = (targets_all_emb * res_tgt.unsqueeze(-1)).sum(1)
        st = res_tgt.sum(1)
        st = st.masked_fill(st == 0, 1)
        tgt_rep = tgt_rep / (st.unsqueeze(1))

        negatives_slice_onehot = self.slice_items[negatives.squeeze()]
        col_indices = torch.arange(self.num_slices - 1).unsqueeze(0)
        clip = (col_indices <= tgt_slices).long().to(self.device)
        res_ngt = clip * negatives_slice_onehot
        negatives_all_emb = item_all_embs[negatives.squeeze()]
        ngt_rep = (negatives_all_emb * res_ngt.unsqueeze(-1)).sum(1)
        sn = res_ngt.sum(1)
        sn = sn.masked_fill(sn == 0, 1)
        ngt_rep = ngt_rep / (sn.unsqueeze(1))
        M = self.log2feats(log_seqs, seq_rep)
        M = M[:, -1, :]

        P = (M * tgt_rep).sum(-1)
        P_neg = (M * ngt_rep).sum(-1)

        comm_size_addon = comm_size[tgt_slices.squeeze(1)].unsqueeze(-1).float()

        trends_item = trends_item.float()
        attn_mtx_pos = self.sftmx((torch.bmm(self.W1(tgt_rep.unsqueeze(1)),
                                             self.W2(trends_item[tgt_slices.squeeze(1)]).transpose(1,
                                                                                    2)).squeeze() / self.scale))
        attn_trend_pos = torch.bmm(attn_mtx_pos.unsqueeze(1),
                                   comm_size_addon * trends_item[tgt_slices.squeeze(1)]).squeeze()
        targets_cluster_cosine_similarity = self.sigmoid(F.cosine_similarity(tgt_rep, attn_trend_pos, dim=1))

        attn_mtx_neg = self.sftmx((torch.bmm(self.W1(ngt_rep.unsqueeze(1)),
                                             self.W2(trends_item[tgt_slices.squeeze(1)]).transpose(1,
                                                                                    2)).squeeze() / self.scale))
        attn_trend_neg = torch.bmm(attn_mtx_neg.unsqueeze(1),
                                   comm_size_addon * trends_item[tgt_slices.squeeze(1)]).squeeze()
        negatives_cluster_cosine_similarity = self.sigmoid(F.cosine_similarity(ngt_rep, attn_trend_neg, dim=1))
        pos_logits = (self.elu(P)+1) * targets_cluster_cosine_similarity
        neg_logits = (self.elu(P_neg)+1) * negatives_cluster_cosine_similarity
        maxi = torch.log(self.sigmoid(pos_logits - neg_logits) + 1e-10)
        loss = -maxi.mean()

        return loss

    def predict(self, trends_item, log_seqs, candidates, log_slices, comm_size):

        item_all_embs = []
        for i in range(self.num_slices - 1):
            item_emb = self.SASs[i].item_embedding.weight
            item_all_embs.append(item_emb)
        item_all_embs = torch.stack(item_all_embs, dim=1)

        log_slice_onehot = self.slice_items[log_seqs]
        flag_onehot = torch.nn.functional.one_hot(log_slices,
                                                  num_classes=self.num_slices - 1)
        col_indices = torch.arange(flag_onehot.size(2)).unsqueeze(0).unsqueeze(0)
        clip = (col_indices <= log_slices.unsqueeze(-1)).long().to(
            self.device)
        res = clip * log_slice_onehot

        seq_all_emb = item_all_embs[log_seqs]
        res = res.unsqueeze(-1)
        masked_m = seq_all_emb * res
        sum_masked_m = torch.sum(masked_m, dim=2)
        sq = torch.sum(res, dim=2)
        sq = sq.masked_fill(sq == 0, 1)
        seq_rep = sum_masked_m / sq
        candidates_slice_onehot = self.slice_items[candidates]
        candidates_all_emb = item_all_embs[candidates]
        cdd_rep = (candidates_all_emb * candidates_slice_onehot.unsqueeze(-1)).sum(2)
        scdd = candidates_slice_onehot.sum(2)
        scdd = scdd.masked_fill(scdd == 0, 1)
        cdd_rep = cdd_rep / (scdd.unsqueeze(-1))
        M = self.log2feats_eval(log_seqs, seq_rep)
        M = M[:, -1, :]
        prediction = (M.unsqueeze(1) * cdd_rep).sum(-1)

        trends_item = trends_item.float()
        tendency_list = []
        Cost_list = []
        for i in range(trends_item.size(0)):
            tendency_list.append(trends_item[i])
            if i == 0:
                continue
            Cost = pairwise_distances(tendency_list[-1].detach().cpu().numpy(),
                                      tendency_list[-2].detach().cpu().numpy(), metric='euclidean')
            Cost_list.append(Cost)
        P_list = []
        for interval_id, Cost in enumerate(Cost_list):
            r = comm_size[interval_id + 1]
            c = comm_size[interval_id]
            P, _ = compute_optimal_transport(Cost, r, c, lam=self.ot_lam, epsilon=1e-6)
            P = P / ((P.sum(1) + (1e-10))[:, np.newaxis])
            P_list.append(P)

        C1_longterm = self.alpha * (P_list[0] @ trends_item[0].detach().cpu().numpy()) + (1 - self.alpha) * (
            trends_item[1].detach().cpu().numpy())
        C2_longterm = self.alpha * (P_list[1] @ C1_longterm) + (1 - self.alpha) * (
            trends_item[2].detach().cpu().numpy())
        C3_longterm = self.alpha * (P_list[2] @ C2_longterm) + (1 - self.alpha) * (
            trends_item[3].detach().cpu().numpy())
        C4_longterm = self.alpha * (P_list[3] @ C3_longterm) + (1 - self.alpha) * (
            trends_item[4].detach().cpu().numpy())

        C4_longterm = torch.from_numpy(C4_longterm).to(self.device)

        attn_eval = self.sftmx_eval((self.W1(cdd_rep) @ (self.W2(C4_longterm).t())) / self.scale)
        attn_trend_eval = attn_eval @ C4_longterm
        cdd_cluster_cosine_similarity = self.sigmoid(
            F.cosine_similarity(cdd_rep, attn_trend_eval, dim=2))
        prediction = (self.elu(prediction)+1) * cdd_cluster_cosine_similarity

        return prediction