import copy
import faiss
import numpy as np

def Move_overlap(pq_centroids, features, pq_code, pq=4, seg_class_num=256):
    assert (seg_class_num ** pq >= len(features))
    dic = {}
    overlab = []
    index = []
    for i in range(pq_code.shape[0]):
        if str(pq_code[i]) not in dic:
            dic[str(pq_code[i])] = 0
        else:
            overlab.append(pq_code[i])
            index.append(i)
    idx = -1

    for item in overlab:
        idx += 1

        feature = features[index[idx]]
        dis_v = []
        sub_len = len(feature) // pq
        for seg in range(pq):
            sub_f = feature[seg * sub_len: (seg + 1) * sub_len]
            sub_c = pq_centroids[seg]
            sub_dis = np.sum(np.square(sub_c - sub_f), axis=1)
            dis_v += list(sub_dis)

        # all dis index from small to large
        dis_v_arg = np.argsort(np.array(dis_v))

        # Op of move overlap
        guard = 1
        count = 0

        while guard:
            for _ in dis_v_arg:
                temp_item = copy.deepcopy(item)
                if str(temp_item) not in dic:
                    guard = 0
                    dic[str(temp_item)] = 0
                    break
            if guard == 1:
                count += 1
                item[int(dis_v_arg[count] // seg_class_num)] = int(dis_v_arg[count] % seg_class_num)
                dis_v_arg = dis_v_arg[count:]
        pq_code[index[idx]] = temp_item
    return pq_code


def get_pq(data_avg, dim, pq, bit, prefix):
    index = faiss.IndexPQ(dim, pq, bit)
    index.train(data_avg)
    index.add(data_avg)

    codes = faiss.vector_to_array(index.codes).reshape(-1, pq)
    centriods = faiss.vector_to_array(index.pq.centroids).reshape(pq, 2 ** bit, -1)

    code_set_len = len(set([str(i) for i in codes]))
    print(f'overlap_{dim} pq{pq}: ', code_set_len)
    if code_set_len != len(data_avg):
        codes = Move_overlap(centriods, data_avg, pq_code=codes, pq=pq, seg_class_num=2 ** bit)
        print(f'not_{dim} overlap pq{pq}: ', len(set([str(i) for i in codes])))
    np.save(f'{prefix}_glint360k_PQ{pq}_nbit{pq * bit}_no_ovelap.npy', codes)

prefix = 'r50'
avg_feature = np.load(f'{prefix}_glint360k_avg_feature.npy')
print('avg_feature_shape: ',avg_feature.shape)
print('total_class_num: ', avg_feature.shape)

dim = avg_feature.shape[1]

pq = 4
bit = 8
get_pq(avg_feature, dim, pq, bit, prefix)
pq = 8
bit = 8
get_pq(avg_feature, dim, pq, bit, prefix)
pq = 16
bit = 8
get_pq(avg_feature, dim, pq, bit, prefix)

print('Done')