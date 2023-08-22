import sys

from torch.utils.data import Dataset

def load_data(path):
    with open(path, 'r', encoding='utf8') as f:
        next(f)
        seq_list = []
        target = []
        seq_list_sliceid = []
        target_sliceid = []
        user_list = []
        for lid, line in enumerate(f):
            line = line.strip()
            item_seq = line.split(':')[1].split('|')
            item_sliceid_dict = {} # {item:sliceid}
            for i, items in enumerate(item_seq):
                if items == '':
                    continue
                items = items.split(',')
                for item in items:
                    item_sliceid_dict[int(item)] = i

            line = line.replace('|', ',')
            uid = int(line.split(':')[0])
            interactions = line.split(':')[1]
            interactions = interactions.split(',')
            interactions =[int(inter) for inter in interactions if inter!='']

            for j in range(1, len(interactions)):
                seq_list.append(interactions[:j][-20:])
                target.append(interactions[j])

                seq_list_sliceid.append([item_sliceid_dict[item] for item in interactions[:j][-20:]])
                target_sliceid.append(item_sliceid_dict[interactions[j]])
                user_list.append(uid)

    return seq_list, target, seq_list_sliceid, target_sliceid, user_list


def load_data_inf(path):
    with open(path, 'r', encoding='utf8') as f:
        next(f)
        seq_list = []
        target = []
        seq_list_sliceid = []
        user_list = []

        for lid, line in enumerate(f):
            line = line.strip()
            item_seq = line.split(':')[1].split('|')
            item_sliceid_dict = {} # {item:sliceid}
            for i, items in enumerate(item_seq):
                if items == '':
                    continue
                items = items.split(',')
                for item in items:
                    item_sliceid_dict[int(item)] = i

                if i == len(item_seq)-1:
                    new_items = [int(item) for item in items]
            uid = int(line.split(':')[0])
            line = line.split(':')[1]

            past = line.split('|')[:-1]
            interactions = []
            for items in past:
                if items == '':
                    continue
                items = items.split(',')
                items = [int(item) for item in items]
                interactions.extend(items)
            interactions = interactions[-20:]
            interactions_sliceid = [item_sliceid_dict[item] for item in interactions]
            for item in new_items:
                seq_list.append(interactions)
                target.append(item)
                user_list.append(uid)
                seq_list_sliceid.append(interactions_sliceid)

    return seq_list, target, seq_list_sliceid, user_list

class Sports(Dataset):
    def __init__(self, mode, args):
        super(Sports, self).__init__()
        self.mode = mode
        self.args = args
        if mode == 'train':
            self.seq_list, self.target, self.seq_list_sliceid, self.target_sliceid, self.user_list = load_data('./data/Sports_6/train_seq.txt')
        if mode == 'val':
            self.seq_list, self.target, self.seq_list_sliceid, self.user_list = load_data_inf('./data/Sports_6/out_val.txt')
        if mode == 'test':
            self.seq_list, self.target, self.seq_list_sliceid, self.user_list = load_data_inf('./data/Sports_6/out_test.txt')

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        if self.mode == 'train':
            uid = self.user_list[index]
            seq = self.seq_list[index]
            tgt = self.target[index]
            seq_sliceid = self.seq_list_sliceid[index]
            tgt_sliceid = self.target_sliceid[index]

            lenth = len(seq)
            pad_seq = [0] * (20-lenth)
            pad_seq.extend(seq)
            pad_seq.append(tgt)
            pad_seq.append(uid)

            pad_seq_sliceid = [0] * (20-lenth)
            pad_seq_sliceid.extend(seq_sliceid)
            pad_seq_sliceid.append(tgt_sliceid)


            return pad_seq, pad_seq_sliceid

        elif self.mode == 'val' or self.mode == 'test':
            uid = self.user_list[index]
            seq = self.seq_list[index]
            tgt = self.target[index]
            seq_sliceid = self.seq_list_sliceid[index]

            lenth = len(seq)
            pad_seq = [0] * (20 - lenth)
            pad_seq.extend(seq)
            pad_seq.append(tgt)
            pad_seq.append(uid)

            pad_seq_sliceid = [0] * (20 - lenth)
            pad_seq_sliceid.extend(seq_sliceid)


            return pad_seq, pad_seq_sliceid


class Phones(Dataset):
    def __init__(self, mode, args):
        super(Phones, self).__init__()
        self.mode = mode
        self.args = args
        if mode == 'train':
            self.seq_list, self.target, self.seq_list_sliceid, self.target_sliceid, self.user_list = load_data('./data/Phones_6/train_seq.txt')
        if mode == 'val':
            self.seq_list, self.target, self.seq_list_sliceid, self.user_list = load_data_inf('./data/Phones_6/out_val.txt')
        if mode == 'test':
            self.seq_list, self.target, self.seq_list_sliceid, self.user_list = load_data_inf('./data/Phones_6/out_test.txt')

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        if self.mode == 'train':
            uid = self.user_list[index]
            seq = self.seq_list[index]
            tgt = self.target[index]
            seq_sliceid = self.seq_list_sliceid[index]
            tgt_sliceid = self.target_sliceid[index]

            lenth = len(seq)
            pad_seq = [0] * (20-lenth)
            pad_seq.extend(seq)
            pad_seq.append(tgt)
            pad_seq.append(uid)

            pad_seq_sliceid = [0] * (20-lenth)
            pad_seq_sliceid.extend(seq_sliceid)
            pad_seq_sliceid.append(tgt_sliceid)


            return pad_seq, pad_seq_sliceid

        elif self.mode == 'val' or self.mode == 'test':
            uid = self.user_list[index]
            seq = self.seq_list[index]
            tgt = self.target[index]
            seq_sliceid = self.seq_list_sliceid[index]

            lenth = len(seq)
            pad_seq = [0] * (20 - lenth)
            pad_seq.extend(seq)
            pad_seq.append(tgt)
            pad_seq.append(uid)

            pad_seq_sliceid = [0] * (20 - lenth)
            pad_seq_sliceid.extend(seq_sliceid)


            return pad_seq, pad_seq_sliceid

class RR(Dataset):
    def __init__(self, mode, args):
        super(RR, self).__init__()
        self.mode = mode
        self.args = args
        if mode == 'train':
            self.seq_list, self.target, self.seq_list_sliceid, self.target_sliceid, self.user_list = load_data('./data/RR_4_sub/train_seq.txt')
        if mode == 'val':
            self.seq_list, self.target, self.seq_list_sliceid, self.user_list = load_data_inf('./data/RR_4_sub/out_val.txt')
        if mode == 'test':
            self.seq_list, self.target, self.seq_list_sliceid, self.user_list = load_data_inf('./data/RR_4_sub/out_test.txt')

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        if self.mode == 'train':
            uid = self.user_list[index]
            seq = self.seq_list[index]
            tgt = self.target[index]
            seq_sliceid = self.seq_list_sliceid[index]
            tgt_sliceid = self.target_sliceid[index]

            lenth = len(seq)
            pad_seq = [0] * (20-lenth)
            pad_seq.extend(seq)
            pad_seq.append(tgt)
            pad_seq.append(uid)

            pad_seq_sliceid = [0] * (20-lenth)
            pad_seq_sliceid.extend(seq_sliceid)
            pad_seq_sliceid.append(tgt_sliceid)



            return pad_seq, pad_seq_sliceid

        elif self.mode == 'val' or self.mode == 'test':
            uid = self.user_list[index]
            seq = self.seq_list[index]
            tgt = self.target[index]
            seq_sliceid = self.seq_list_sliceid[index]

            lenth = len(seq)
            pad_seq = [0] * (20 - lenth)
            pad_seq.extend(seq)
            pad_seq.append(tgt)
            pad_seq.append(uid)

            pad_seq_sliceid = [0] * (20 - lenth)
            pad_seq_sliceid.extend(seq_sliceid)


            return pad_seq, pad_seq_sliceid

class Yelp(Dataset):
    def __init__(self, mode, args):
        super(Yelp, self).__init__()
        self.mode = mode
        self.args = args
        if mode == 'train':
            self.seq_list, self.target, self.seq_list_sliceid, self.target_sliceid, self.user_list = load_data('./data/Yelp_6/train_seq.txt')
        if mode == 'val':
            self.seq_list, self.target, self.seq_list_sliceid, self.user_list = load_data_inf('./data/Yelp_6/out_val.txt')
        if mode == 'test':
            self.seq_list, self.target, self.seq_list_sliceid, self.user_list = load_data_inf('./data/Yelp_6/out_test.txt')

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        if self.mode == 'train':
            uid = self.user_list[index]
            seq = self.seq_list[index]
            tgt = self.target[index]
            seq_sliceid = self.seq_list_sliceid[index]
            tgt_sliceid = self.target_sliceid[index]

            lenth = len(seq)
            pad_seq = [0] * (20-lenth)
            pad_seq.extend(seq)
            pad_seq.append(tgt)
            pad_seq.append(uid)

            pad_seq_sliceid = [0] * (20-lenth)
            pad_seq_sliceid.extend(seq_sliceid)
            pad_seq_sliceid.append(tgt_sliceid)


            return pad_seq, pad_seq_sliceid

        elif self.mode == 'val' or self.mode == 'test':
            uid = self.user_list[index]
            seq = self.seq_list[index]
            tgt = self.target[index]
            seq_sliceid = self.seq_list_sliceid[index]

            lenth = len(seq)
            pad_seq = [0] * (20 - lenth)
            pad_seq.extend(seq)
            pad_seq.append(tgt)
            pad_seq.append(uid)

            pad_seq_sliceid = [0] * (20 - lenth)
            pad_seq_sliceid.extend(seq_sliceid)


            return pad_seq, pad_seq_sliceid
