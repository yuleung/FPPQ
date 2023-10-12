# PQ_search
import os
import sys
import copy
import faiss
import argparse
import numpy as np
from sklearn import preprocessing

current_dir = os.getcwd()
sys.path.append(current_dir)
from utils.utils_config import get_config

Dim = 512
Seg = 4
Bit = 8
Seg_dim = int(Dim / Seg)


def main(args):
    cfg = get_config(args.config)
    feature_root = f'{cfg.output}/feature_saved/'

    p_features = preprocessing.normalize(np.load(feature_root + 'facecrub_data_feature_total_not_nor.npy'))
    p_labels_ori = np.load(feature_root + 'facecrub_label_total.npy').tolist()
    g_features = preprocessing.normalize(np.load(feature_root + 'mage_data_feature_total_not_nor.npy'))

    if 1:
        g_features = preprocessing.normalize(g_features.reshape(-1, Dim // Seg)).reshape(-1, Dim)
        p_features = preprocessing.normalize(p_features.reshape(-1, Dim // Seg)).reshape(-1, Dim)
    feature_dict = {}

    def topk_acc_index(index, topk):
        correct_num_mx = [0] * 4
        index_cut = index[:, :topk]
        for sorted_id in index_cut:
            for k in range(topk):
                if sorted_id[k] - g_features.shape[0] == 0:
                    if k < 1:
                        correct_num_mx[0] += 1
                    if k < 2:
                        correct_num_mx[1] += 1
                    if k < 5:
                        correct_num_mx[2] += 1
                    if k < 20:
                        correct_num_mx[3] += 1
        return correct_num_mx

    for i in range(len(p_labels_ori)):
        if p_labels_ori[i] not in feature_dict:
            feature_dict[p_labels_ori[i]] = [p_features[i]]
        else:
            feature_dict[p_labels_ori[i]] = feature_dict[p_labels_ori[i]] + [p_features[i]]
    total_count = 0
    for i in range(80):
        total_count += len(feature_dict[i]) * (len(feature_dict[i]) - 1)

    index = faiss.IndexPQ(Dim, Seg, Bit)
    index.train(g_features)
    index.add(g_features)

    acc1 = 0
    acc2 = 0
    acc5 = 0
    acc20 = 0
    for classes in range(80):
        search_list = feature_dict[classes]
        correct_num_mx = [0] * 4
        for sample in range(len(search_list)):
            to_add_sample = copy.deepcopy(search_list[sample])
            index.add(np.array([to_add_sample]))
            Q_f = copy.deepcopy(search_list)
            del Q_f[sample]
            _, index_1 = index.search(np.ascontiguousarray(Q_f), 20)
            correct_num_mx = topk_acc_index(index_1, topk=20)
            acc1 = acc1 + correct_num_mx[0]
            acc2 = acc2 + correct_num_mx[1]
            acc5 = acc5 + correct_num_mx[2]
            acc20 = acc20 + correct_num_mx[3]
            index.remove_ids(np.array([len(g_features)]))
        print('Done of PQ {} at class {}'.format(Seg, classes))

    print('top1: ', acc1 / total_count)
    print('top2: ', acc2 / total_count)
    print('top5: ', acc5 / total_count)
    print('top20: ', acc20 / total_count)

    # save result
    save_path = f'{cfg.output}/result'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(f'{save_path}/log_ce_pq_search_Fn.txt', 'a+') as f:
        f.write(f'Seg:   {Seg}\n')
        f.write(f'top1:  {acc1 / total_count}\n')
        f.write(f'top2:  {acc2 / total_count}\n')
        f.write(f'top5:  {acc5 / total_count}\n')
        f.write(f'top20: {acc20 / total_count}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())