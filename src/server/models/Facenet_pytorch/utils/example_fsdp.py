
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from src.server.models.Facenet_pytorch.inception_resnet_v1 import InceptionResnetV1
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.multiprocessing as mp
import os
import functools

from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)


my_auto_wrap_policy = functools.partial(
        							size_based_auto_wrap_policy, 
        							min_num_params=20000)

def setup(rank:int, world_size:int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_mapping(rank, world_size, batch):
        setup(rank, world_size)
        mod = InceptionResnetV1(pretrained = 'casia-webface', 
								classify=False,
								num_classes=None, 
								dropout_prob=0.6,
								device = rank,
								pretrained_weight_dir = '/kaggle/input/massive-faces/Facenet_pytorch/Facenet_pytorch'
        					).to(rank)
        
        opt_mod = FSDP(mod, use_orig_params = True, auto_wrap_policy= my_auto_wrap_policy)
        opt_mod = torch.compile(opt_mod)
        t = torch.randn(2,3,160,160).to(rank)
        print(opt_mod(t))
        cleanup()
