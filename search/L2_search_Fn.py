# L2_search
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


def main(args):
    cfg = get_config(args.config)
    feature_root = f'{cfg.output}/feature_saved/'
    p_features = preprocessing.normalize(np.load(feature_root + 'facecrub_data_feature_total_not_nor.npy'))
    p_labels_ori = np.load(feature_root + 'facecrub_label_total.npy').tolist()
    g_features = preprocessing.normalize(np.load(feature_root + 'mage_data_feature_total_not_nor.npy'))

    feature_dict = {}
    for i in range(len(p_labels_ori)):
        if p_labels_ori[i] not in feature_dict:
            feature_dict[p_labels_ori[i]] = [p_features[i]]
        else:
            feature_dict[p_labels_ori[i]] = feature_dict[p_labels_ori[i]] + [p_features[i]]

    def topk_acc_index(index, topk):
        correct_num_mx = [0] * 4
        index_cut = index[:, :topk]

        for sorted_id in index_cut:  # each reslut
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

    acc1 = [0]
    acc2 = [0]
    acc5 = [0]
    acc20 = [0]

    total_count = 0
    for i in range(80):
        total_count += len(feature_dict[i]) * (len(feature_dict[i]) - 1)

    L2 = faiss.IndexFlatL2(512)
    L2.add(g_features)
    for classes in range(80):
        search_list = feature_dict[classes]
        correct_num_mx = [0] * 4
        for sample in range(len(search_list)):
            to_add_sample = copy.deepcopy(search_list[sample])
            L2.add(np.array([to_add_sample]))
            Q_f = copy.deepcopy(search_list)
            del Q_f[sample]
            _, index_1 = L2.search(np.ascontiguousarray(Q_f), 20)
            correct_num_mx = topk_acc_index(index_1, topk=20)
            acc1[0] = acc1[0] + correct_num_mx[0]
            acc2[0] = acc2[0] + correct_num_mx[1]
            acc5[0] = acc5[0] + correct_num_mx[2]
            acc20[0] = acc20[0] + correct_num_mx[3]
            L2.remove_ids(np.array([len(g_features)]))
        print('Done of L2 at class {}'.format(classes))

    acc1_avg = np.array(acc1)[0] / total_count
    acc2_avg = np.array(acc2)[0] / total_count
    acc5_avg = np.array(acc5)[0] / total_count
    acc20_avg = np.array(acc20)[0] / total_count

    print(acc1_avg)
    print(acc2_avg)
    print(acc5_avg)
    print(acc20_avg)

    save_path = f'{cfg.output}/result'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(f'{save_path}/log_pq_L2_Fn.txt', 'a+') as f:
        f.write('nor_L2: \n')
        f.write('top1:  ' + str(acc1_avg) + '\n')
        f.write('top2:  ' + str(acc2_avg) + '\n')
        f.write('top5:  ' + str(acc5_avg) + '\n')
        f.write('top20: ' + str(acc20_avg) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())