import numpy as np
from sklearn import preprocessing

data = np.load('r50_glint360k_data_feature_total_nor.npy')
o_label = np.load('r50_glint360k_label_total.npy')

print('loaded')
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
np.save('r50_glint360k_avg_feature.npy', data_avg)

