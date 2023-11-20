#1、prepare
#get pretrained model
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=222222 train.py  configs/glint360k_r50.py
#get PQ_label
python get_feature/training_feature/get_feature.py configs/glint360k_r50.py
python get_feature/training_feature/get_pq_label.py --prefix='r50'

#2、train model
#train model using our FPPQ
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=222222 train_FPPQ.py  configs/glint360k_r50_pq4.py


#3、eval
#eval in unseen setting
#get query feature
python get_feature/training_feature/get_facescrub_feature_FPPQ.py configs/glint360k_r50_pq4.py
#get gallery feature
python get_feature/training_feature/get_megaface_feature_FPPQ.py configs/glint360k_r50_pq4.py
#search
#L2 search
python search/L2_search_Fn.py configs/glint360k_r50_FPPQ.py
#PQ search with learned codebook
python search/PQ_search_with_learnedCodebook.py configs/glint360k_r50_FPPQ.py --pq=4
