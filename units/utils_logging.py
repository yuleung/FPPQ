import importlib
import os.path as osp


def get_config(config_file):
    assert config_file.startswith('configs/') or config_file.startswith('./configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.base")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    return cfgliangyu@AI-Internal-99137:/data2/liangyu/arcface_torch_repeat_ly/utils$
liangyu@AI-Internal-99137:/data2/liangyu/arcface_torch_repeat_ly/utils$
liangyu@AI-Internal-99137:/data2/liangyu/arcface_torch_repeat_ly/utils$ cat utils_config.py
import importlib
import os.path as osp


def get_config(config_file):
    assert config_file.startswith('configs/') or config_file.startswith('./configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.base")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    return cfgliangyu@AI-Internal-99137:/data2/liangyu/arcface_torch_repeat_ly/utils$ cat utils_logging.py
import os
import sys
import logging


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logging(rank, models_root):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)
