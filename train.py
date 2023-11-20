import argparse
import logging
import os

import numpy as np
import torch
from torch import distributed
import losses
from backbones import get_model
from dataset import get_dataloader
from torch.utils.data import DataLoader
from lr_scheduler import PolyScheduler
from partial_fc import PartialFC
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
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
    cfg.output = cfg.output + "/" + "training"
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)
    num_image, train_loader = get_dataloader(
        cfg.rec, local_rank=args.local_rank, batch_size=cfg.batch_size, dali=cfg.dali)

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size, pq=cfg.pq, Wnor=cfg.Fnor,
        Fnor=cfg.Fnor).cuda()
    if rank == 0:
        print(backbone)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.train()

    total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // total_batch_size * cfg.num_epoch

    margin_loss = losses.get_loss(cfg.loss)
    module_partial_fc = PartialFC(
        margin_loss, cfg.embedding_size, cfg.num_classes,
        cfg.sample_rate, cfg.fp16)
    module_partial_fc.train().cuda()

    opt = torch.optim.SGD(
        params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
        lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step
    )
    start_epoch = 0
    global_step = 0

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=None
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):

            global_step += 1
            local_embeddings, outs = backbone(img)
            loss = cfg.theta1 * module_partial_fc(local_embeddings, local_labels, opt)

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)


                if global_step % cfg.verbose == 0 and global_step > 200:
                    callback_verification(global_step, backbone)

        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "state_dict_backbone": backbone.module.state_dict(),
            "state_dict_softmax_fc": module_partial_fc.state_dict(),
            "state_optimizer": opt.state_dict(),
            "state_lr_scheduler": lr_scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        if cfg.dali:
            train_loader.reset()

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
