import os
import sys
import torch
import logging
import argparse
import numpy as np
from torch import distributed
import torch.nn.functional as F
from sklearn import preprocessing
from torch.utils.data import DataLoader

from backbones import new_get_model
from dataset import get_dataloader
from utils.utils_config import get_config


try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12561",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    bs = 2048
    seed = 2333
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.cuda.set_device(args.local_rank)
    cfg = get_config(args.config)
    os.makedirs(cfg.output, exist_ok=True)
    _, train_loader = get_dataloader(cfg.rec, local_rank=args.local_rank, batch_size=bs, drop_last=False,
                                     shuffle=False, dali=cfg.dali)

    backbone = new_get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, pq=cfg.pq, num_features=cfg.embedding_size).cuda()
    backbone_pth = os.path.join(f"{cfg.output}/training", "model.pt")
    backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(0)))
    print('loaded!')
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.eval()

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    save_path = './feature_save/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    global_step = 0
    epoch_label = []
    epoch_feature = []
    with torch.no_grad():
        for epoch in range(1):
            if isinstance(train_loader, DataLoader):
                train_loader.sampler.set_epoch(epoch)
            for step, (img, label) in enumerate(train_loader):
                global_step += 1
                local_embeddings, _ = backbone(img)
                features = F.normalize(local_embeddings)
                feature = features.detach().cpu().numpy()
                epoch_feature.append(feature)
                epoch_label.append(label.detach().cpu().numpy())
                print('step:', step * bs)

        feature_total = np.array(list(np.vstack(np.array(epoch_feature[:-1]))) + list(np.array(epoch_feature[-1])))
        print('feature_total.shape:', feature_total.shape)
        np.save(save_path + f'./{cfg.network}_glint360k_data_feature_total_nor.npy', preprocessing.normalize(feature_total))

        label_total = np.array(list(np.hstack(np.array(epoch_label[:-1]))) + list(np.array(epoch_label[-1])))
        print('label_total.shape:', label_total.shape)
        np.save(save_path + f'./{cfg.network}_glint360k_label_total.npy', label_total)
    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())