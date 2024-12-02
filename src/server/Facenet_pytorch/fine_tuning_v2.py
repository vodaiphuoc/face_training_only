"""
This module use DDP rather than FSDP
"""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from typing import List, Dict, Tuple, Any, Union
import itertools
import functools
import random
from copy import deepcopy
from tqdm import tqdm
import cv2 as cv
import json
from src.server.Facenet_pytorch.utils.fine_tuning_utils import TripLetDataset_V2, CustomeTripletLoss
from src.server.Facenet_pytorch.inception_resnet_v1 import InceptionResnetV1

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import os

def setup(rank:int, world_size:int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



class FineTuner(object):

	freeze_list = ['conv2d_4a', 'conv2d_4b', 'repeat_1','mixed_6a','repeat_2','mixed_7a','repeat_3', 
					'block8', 'avgpool_1a', 'last_linear', 'last_bn'
					]

	def __init__(self, 
				rank:int,
				world_size:int,
				num_epochs:int,
				gradient_accumulate_steps: int,
				lr: float,
				pretrained_weight_dir: str = 'src\\server\\models\\pretrained_weights\\Facenet_pytorch',
				return_examples:int = 512,
				data_folder_path:str = 'face_dataset/faces_only',
				number_other_users:float = 0.2,
				p_n_ratio:int = 4,
				number_celeb_in_train:int = 500,
				batch_size:int = 64,
				num_workers:int = 2
				):
		self.num_epochs = num_epochs
		self.gradient_accumulate_steps = gradient_accumulate_steps
		self.lr = lr
		self.master_batch_size = batch_size*return_examples
		self.p_n_ratio = p_n_ratio

		self.loader_args_dict = {
			'return_examples': return_examples,
			'data_folder_path': data_folder_path,
			'number_other_users': number_other_users,
			'p_n_ratio': p_n_ratio,
			'number_celeb_in_train': number_celeb_in_train,
			'batch_size': batch_size,
			'num_workers': num_workers
		}

		my_auto_wrap_policy = functools.partial(
        							size_based_auto_wrap_policy, 
        							min_num_params=25000)

		model = InceptionResnetV1(pretrained = 'casia-webface', 
								classify=False,
								num_classes=None, 
								dropout_prob=0.6,
								device = rank,
								pretrained_weight_dir = pretrained_weight_dir)

		for name, module in model.named_modules():
			if name not in self.freeze_list:
				for param in module.parameters():
					param.requires_grad = False
			else:
				for param in module.parameters():
					param.requires_grad = True

		# import torch._dynamo
		# torch._dynamo.reset()
		# import triton
		
		torch.cuda.set_device(rank)
		
		model = model.to(rank)
		ddp_model = DDP(model, device_ids=[rank])
		
		self.model = torch.compile(ddp_model,
						mode="reduce-overhead",
						fullgraph = False)

		local_loader_args_dict = deepcopy(self.loader_args_dict)
		local_loader_args_dict['rank'] = rank
		local_loader_args_dict['world_size'] = world_size

		self.train_loader, self.train_sampler = FineTuner._make_loaders(is_train = True,**local_loader_args_dict)
		self.val_loader, self.val_sampler = FineTuner._make_loaders(is_train = False,**local_loader_args_dict)


		self.optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.lr)
		self.scheduler = MultiStepLR(self.optimizer, milestones = [i*self.num_epochs//3 for i in range(1,3)])

		# self.loss_fn = torch.nn.TripletMarginWithDistanceLoss(margin = 0.9, 
		# 					distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y),
		# 					swap=False,
		# 					reduction='mean')

		loss_fn = CustomeTripletLoss(margin = 1.0, 
			device = rank, 
			batch_size = batch_size, 
			return_examples = return_examples,
			p_n_ratio = p_n_ratio
			)
		self.loss_fn = torch.compile(loss_fn.to(rank), 
			options = {'triton.cudagraphs': True}, 
			fullgraph = True)
	
	@staticmethod
	def _make_loaders(is_train:bool,
					rank:int,
					world_size:int,
					return_examples:int,
					data_folder_path:str,
					number_other_users:float,
					p_n_ratio: int,
					number_celeb_in_train:int,
					batch_size:int,
					num_workers:int
					):
		dataset = TripLetDataset_V2(return_examples = return_examples,
								is_train = is_train,
								data_folder_path = data_folder_path,
								number_other_users = number_other_users,
								p_n_ratio = p_n_ratio,
								number_celeb_in_train = number_celeb_in_train
		)
		sampler = DistributedSampler(dataset, 
									rank=rank, 
									num_replicas=world_size, 
									shuffle=True if is_train else False)
		
		return (torch.utils.data.DataLoader(dataset, 
										batch_size= batch_size,
										shuffle=False, 
										sampler = sampler,
										num_workers=num_workers,
										pin_memory=True, 
										drop_last=True,
										prefetch_factor=2,
										persistent_workers=True
		),
		sampler)

	def _pre_process_batch_data(self, 
								list_batch: List[torch.Tensor], 
								rank: Union[int, torch.device]
								)->torch.Tensor:

		return torch.cat([ batch.reshape((self.master_batch_size, 3, 160, 160)) 
							if ith != len(list_batch)-1
							else
							batch.reshape((self.master_batch_size*self.p_n_ratio, 3, 160, 160)) 
							for ith, batch in enumerate(list_batch)
						],
						dim = 0
						).to(rank)

	def _train(self, rank:int, world_size:int)->float:
		ddp_loss = torch.zeros(2).to(rank)
		self.model.train()
		print('Length of loader',len(self.train_loader))
		for batch_idx, (a_batch, p_batch, n_batch) in tqdm(enumerate(self.train_loader),
															total = len(self.train_loader)):

			model_inputs = self._pre_process_batch_data([a_batch, p_batch, n_batch], rank)
			embeddings = self.model(model_inputs)

			a_embeddings = embeddings[0: self.master_batch_size,:]
			p_embeddings = embeddings[self.master_batch_size: 2*self.master_batch_size,:]
			n_embeddings = embeddings[2*self.master_batch_size:,:]

			loss = self.loss_fn(a_embeddings, p_embeddings, n_embeddings)

			loss = loss/self.gradient_accumulate_steps
			loss.backward()

			ddp_loss[0] += loss.item()
			ddp_loss[1] += len(model_inputs)

			if ((batch_idx + 1) % self.gradient_accumulate_steps == 0) or \
				(batch_idx + 1 == len(self.train_loader)):

				self.optimizer.step()
				self.optimizer.zero_grad()

		dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
		return ddp_loss[0]/ddp_loss[1]

	def _eval(self, rank:int, world_size:int)->float:
		ddp_loss = torch.zeros(2).to(rank)
		self.model.eval()
		print('Length of loader',len(self.val_loader))
		with torch.no_grad():
			for batch_idx, (val_a_batch, val_p_batch, val_n_batch) in enumerate(self.val_loader):
				val_model_inputs = self._pre_process_batch_data([val_a_batch, val_p_batch, val_n_batch], rank)

				embeddings = self.model(val_model_inputs)

				a_embeddings = embeddings[0: self.master_batch_size,:]
				p_embeddings = embeddings[self.master_batch_size: 2*self.master_batch_size,:]
				n_embeddings = embeddings[2*self.master_batch_size:,:]

				ddp_loss[0] += self.loss_fn(a_embeddings, p_embeddings, n_embeddings).item()
				ddp_loss[1] += len(val_model_inputs)

		dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
		return ddp_loss[0]/ddp_loss[1]

def training_loop( rank :int,
					world_size:int,
					trainer_args: dict, 
					save_path: str):
	"""Main training function
		In distributed mode, calling the set_epoch() method at the beginning of each epoch before 
	creating the DataLoader iterator is necessary to make shuffling work properly across multiple 
	epochs. Otherwise, the same ordering will be always used.
	"""
	setup(rank, world_size)
	
	trainer_args['rank'] = rank
	trainer_args['world_size'] = world_size
	trainer = FineTuner(**trainer_args)

	train_logs = {}
	val_logs = {}

	for epoch in range(1,trainer.num_epochs+1):
		trainer.train_sampler.set_epoch(epoch)
		train_logs[f'Epoch_{epoch}'] = trainer._train(rank, world_size)

		if trainer.num_epochs//epoch == 2 or epoch == trainer.num_epochs:
			trainer.val_sampler.set_epoch(epoch)
			val_logs[f'Epoch_{epoch}'] = trainer._eval(rank, world_size)

		trainer.scheduler.step()

	if rank == 0:
		print(train_logs)
		print(val_logs)

	dist.barrier()
	state = trainer.model.state_dict()
	# save checkpoints
	if rank == 0:
		torch.save(state, save_path)

	cleanup()