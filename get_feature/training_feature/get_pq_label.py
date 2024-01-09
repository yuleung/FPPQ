from sklearn import preprocessing
import copy
import faiss
import numpy as np
import argparse

def get_avg_feature(data, o_label):
    sort_index = np.argsort(o_label)
    print('sorted')
    features = data[sort_index]
    labels = o_label[sort_index]
    print(labels.shape)

    avg_features = []
    guard = 0
    count_pre = 0
    count_last = 0
    count = 0
    for label in labels:
        if label != guard and count_last == len(labels)-1:
            assert len(set(labels[count_pre:count_last]))==1
            count += count_last-count_pre+1
            avg_feature = np.sum(features[count_pre:count_last],axis=0)/(count_last-count_pre)
            avg_feature = avg_feature.reshape(1,-1)
            avg_features.append(avg_feature)
            avg_feature = features[-1].reshape(1,-1)
            avg_features.append(avg_feature)
        elif count_last == len(labels)-1:
            assert len(set(labels[count_pre:count_last+1]))==1
            count += count_last-count_pre+1
            avg_feature = np.sum(features[count_pre:count_last+1],axis=0)/(count_last-count_pre)
            avg_feature = avg_feature.reshape(1,-1)
            avg_features.append(avg_feature)
        elif label != guard:
            assert len(set(labels[count_pre:count_last]))==1
            avg_feature = np.sum(features[count_pre:count_last],axis=0)/(count_last-count_pre)
            count += count_last-count_pre
            avg_feature = avg_feature.reshape(1,-1)
            avg_features.append(avg_feature)
            count_pre = count_last
            guard = label
        count_last += 1

    data_avg = preprocessing.normalize(np.squeeze(avg_features))
    assert(count == len(labels))
    return data_avg

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

        #Calculate the distance between seg-th sub-feature and seg-th codebook
        for seg in range(pq):
            sub_f = feature[seg * sub_len: (seg + 1) * sub_len]
            sub_c = pq_centroids[seg]
            sub_dis = np.sum(np.square(sub_c - sub_f), axis=1)
            dis_v += list(sub_dis)

        # All distance index from small to large
        dis_v_arg = np.argsort(np.array(dis_v))

        # Op of move overlap
        guard = 1

        #We can use the queue to generate the next pq-code to be selected in the form of a generator.  
        #This code changes the encoding of at most two segments, but can be applied to almost any situation
        Queue_template = []                                                                                                                                         
        for i in range(len(dis_v_arg)-1):                                                                                                                           
            Queue_template.append([i, i+1])                                                                                                                         
        Queue = Queue_template                                                                                                                                                                                                                                                                            
        while guard:                                                                                                                                                
            #pop the index has min dis                                                                                                                              
            min_dis_index = Queue.pop(0)                                                                                                                            
            ind1 = dis_v_arg[min_dis_index[0]]                                                                                                                      
            ind2 = dis_v_arg[min_dis_index[1]]                                                                                                                      
            temp_item = copy.deepcopy(item)                                                                                                                         
            temp_item[int(ind1//seg_class_num)] = int(ind1 % seg_class_num)                                                                                         
            temp_item[int(ind2//seg_class_num)] = int(ind2 % seg_class_num)                                                                                         
            if str(temp_item) not in dic:                                                                                                                           
                #print(f'1temp_item: {temp_item}')                                                                                                                  
                dic[str(temp_item)] = 0                                                                                                                             
                break                                                                                                                                               
            nex_item_index1 = min_dis_index[0]                                                                                                                      
            if min_dis_index[1] != seg_class_num - 1:                                                                                                                         
                nex_item_index2 = min_dis_index[1] + 1                                                                                                              
                                                                                                                                                                    
            #next index pair insert to Queue                                                                                                                        
            nex_item_dis = dis_v[nex_item_index1] + dis_v[nex_item_index2]                                                                                          
            for Q in range(len(Queue)):                                                                                                                             
                if nex_item_dis < dis_v[dis_v_arg[Queue[Q][0]]] + dis_v[dis_v_arg[Queue[Q][1]]]:                                                                    
                    Queue = Queue[:Q] + [[nex_item_index1, nex_item_index2]] + Queue[Q:]                                                                            
                    break                                                                                                                                                                                                                                                                                  
        #print(temp_item)                                                                                                                                           
        pq_code[index[idx]] = temp_item                                                                                                                             
    return pq_code

def get_pq(data_avg, dim, pq, bit):
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
    return codes

def main(args):
    prefix = args.prefix
    save_path = './feature_save/'
    data = np.load(save_path + f'{prefix}_glint360k_data_feature_total_nor.npy')
    o_label = np.load(save_path + f'{prefix}_glint360k_label_total.npy')

    avg_feature = get_avg_feature(data, o_label)
    np.save(f'{save_path} + {prefix}_glint360k_avg_feature.npy', avg_feature)
    print('feature average done!')
    print('avg_feature_shape: ', avg_feature.shape)
    print('total_class_num: ', avg_feature.shape)

    dim = avg_feature.shape[1]

    pq = 4
    bit = 8
    pq4_code = get_pq(avg_feature, dim, pq, bit)
    np.save(save_path + f'{prefix}_glint360k_PQ{pq}_nbit{pq * bit}_no_ovelap.npy', pq4_code)
    print('PQ4 Label, Done!')
    pq = 8
    bit = 8
    pq8_code = get_pq(avg_feature, dim, pq, bit)
    np.save(save_path + f'{prefix}_glint360k_PQ{pq}_nbit{pq * bit}_no_ovelap.npy', pq8_code)
    print('PQ8 Label, Done!')
    pq = 16
    bit = 8
    pq16_code = get_pq(avg_feature, dim, pq, bit)
    np.save(save_path + f'{prefix}_glint360k_PQ{pq}_nbit{pq * bit}_no_ovelap.npy', pq16_code)
    print('PQ16 Label, Done!')
    print('PQ Label Got! ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get_pq_label")
    parser.add_argument("--prefix", type=str, default='r50', help="backbone")
    main(parser.parse_args())

