import os
import sys
import argparse
import logging
import torch
import numpy as np
from PIL import Image
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F

current_dir = os.getcwd()
sys.path.append(current_dir)

from dataset import DataLoaderX
from backbones import get_model
from utils.utils_config import get_config
from utils.utils_logging import init_logging


class My_Dataset(Dataset):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        with open('./dataset/megaface/megaface_lst_noises_removed.txt', 'r') as f:
            self.filename = f.readlines()
            self.filename = [i.strip() for i in self.filename]

    def __getitem__(self, index):
        self.dataset_path = './dataset/megaface/mega_test_pack_unzip/megaface_images/'
        filename = self.filename[index]
        img = np.array(Image.open(self.dataset_path + filename))
        sample = self.transform(img)
        return sample

    def __len__(self):
        return len(self.filename)


try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12532",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    seed = 2333
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.cuda.set_device(args.local_rank)

    cfg = get_config(args.config)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    train_set = My_Dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=False)
    train_loader = DataLoaderX(local_rank=0, dataset=train_set, batch_size=50, sampler=train_sampler, num_workers=8,
                               pin_memory=True, drop_last=False)

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size, pq=cfg.pq, Wnor=cfg.Fnor,
        Fnor=cfg.Fnor).cuda()

    backbone = backbone.cuda()
    print(backbone)
    backbone_pth = f"{cfg.output}/training/model.pt"
    print(backbone_pth)
    backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(0)))  # ['state_dict_backbone'])
    print('backbone loaded!')
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.eval()

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    save_path = cfg.output + '/feature_saved/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    global_step = 0
    epoch_feature = []
    for epoch in range(1):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for step, img in enumerate(train_loader):
            global_step += 1

            features, y = backbone(img)
            epoch_feature.append(features.detach().cpu().numpy())
            print('step:', step * 50)

        feature_total = np.array(list(np.vstack(np.array(epoch_feature[:-1]))) + list(np.array(epoch_feature[-1])))
        print('feature_total.shape:', feature_total.shape)
        np.save(save_path + 'mage_data_feature_total_not_nor.npy', feature_total)

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())