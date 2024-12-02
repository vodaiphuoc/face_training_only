import torch
import glob
from typing import List, Dict, Tuple, Any, Union
import itertools
import random
from tqdm import tqdm
import cv2 as cv
import json
import logging


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

class TripLetDataset_V2(torch.utils.data.Dataset):
	"""For train dataset, all data, 
	For validation set, collected users are combined with 100 celeb users
	"""
	def __init__(self, 
				is_train = True,
				return_examples:str = 512,
				data_folder_path:str = 'face_dataset\\faces_only', 
				number_other_users:int = 3,
				p_n_ratio = 4,
				number_celeb_in_train:int = 500
				)->None:

		(
			self.glob_iter,
			self.index_iter,
			self.user2img_path
		) = TripLetDataset_V2._make_index_list(is_train = is_train,
											data_folder_path = data_folder_path, 
											number_other_users = number_other_users,
											number_celeb_in_train = number_celeb_in_train
											)
		self.data_folder_path = data_folder_path
		self.return_examples = return_examples
		self.p_n_ratio = p_n_ratio
		logging.basicConfig(level=logging.INFO)
		return None

	@staticmethod
	def _make_index_list(is_train: bool,
						data_folder_path: str,
						number_other_users: int = 3,
						number_celeb_in_train: int = 500
						)->Tuple[List[str], \
								List[Dict[str, Union[int, List[int]]]], \
								List[Dict[str, List[str]]] \
								]:
		# user folders
		if is_train:
			glob_iter = glob.glob("*_*",root_dir = f"{data_folder_path}") + \
						random.sample(glob.glob("[0-9]*", root_dir = f"{data_folder_path}"), 
									k = number_celeb_in_train)

			with open('dataset.json', "w") as f:
				json.dump([{ith:ele} for ith, ele in enumerate(glob_iter)], f, indent = 4)

		else:
			glob_iter = glob.glob("*_*",root_dir = f"{data_folder_path}")  + \
						random.sample(glob.glob("[0-9]*", root_dir = f"{data_folder_path}"), 
									k = number_celeb_in_train)
		
		# MAIN DIFFERENCE IN V2
		# user maps to other users
		# this is main index for iterator
		userIdx2other_usersIdx: List[Dict[str, Union[int, List[int]]]] = []
		for i in range(len(glob_iter)):
			relations = [
					{
						'user_dir_idx': i,
						'other_dir_idx_list': [_i for _i in range(_i, _i+number_other_users)]
					}
					for _i in range(0, len(glob_iter), number_other_users)
					if _i + number_other_users < len(glob_iter)-1 and \
						not (i >= _i and i < _i+number_other_users)
			]
			userIdx2other_usersIdx.extend(relations)

		# user maps to its image paths
		user2img_path = {
			user_dir_idx: [
				_path for _path in glob.glob('*.*',
									root_dir = f"{data_folder_path+'/'+glob_iter[user_dir_idx]}")
			]
			for user_dir_idx in range(len(glob_iter))
		}

		return glob_iter, userIdx2other_usersIdx, user2img_path
		

	def __len__(self):
		return len(self.index_iter)

	@staticmethod
	def _adjust2fixe_size(data_list: List[Dict[str,Any]], 
						num_limit_samples:int
						)-> List[Dict[str,Any]]:

		if len(data_list) >= num_limit_samples:
			# logging.info(f'Trimming...{len(data_list)},{num_limit_samples}')
			return random.sample(data_list, k = num_limit_samples)
		else:
			# logging.info(f'Append...{len(data_list)}')
			ratio = num_limit_samples//len(data_list) + 1
			data_list = data_list*ratio
			return data_list[:num_limit_samples]

	def _get_triplet_index(self, 
							user_dir_idx: int, 
							other_dir_idx_list: List[int]
							)-> Tuple[List[Dict[str, str]]]:
		"""
		Given a list of index (user folder dir)
		return mapping from user anchor/positive files with
		negative files
		"""
		# get total images of current user
		anchor_imgs_path = self.user2img_path[user_dir_idx]
		positives = [
						{user_dir_idx:file_name_pair}
						for file_name_pair \
						in itertools.combinations(anchor_imgs_path,2)
					]
		
		neg_img_list = []
		for other_user_idx in other_dir_idx_list:
			if len(neg_img_list) > int(self.p_n_ratio*self.return_examples)+1:
				break
			else:
				neg_img_list.extend([
										{other_user_idx: img_file_name} 
										for img_file_name in 
										random.sample(self.user2img_path[other_user_idx], 
													k = len(self.user2img_path[other_user_idx])//2)
									])
		return (TripLetDataset_V2._adjust2fixe_size(positives, self.return_examples),
				TripLetDataset_V2._adjust2fixe_size(neg_img_list, self.return_examples*self.p_n_ratio)
				)

	def _paths2tensor(self, path_list: List[str])->torch.Tensor:
		return_tensor = []
		for _path in path_list:
			image = cv.imread(_path)
			image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
			image_tensor = torch.tensor(image).permute(2,0,1)
			return_tensor.append(image_tensor)

		return fixed_image_standardization(torch.stack(return_tensor)).to(torch.float32)

	def __getitem__(self, index:int)->Tuple[torch.Tensor]:
		"""
		Return:
			- Tuple of tensor shape (self.return_examples,3,160,160)
		"""
		usr_other_usr_dict = self.index_iter[index]
		anchor_positives, neg_img_list = self._get_triplet_index(**usr_other_usr_dict)

		a_path, p_path, n_path = [], [], []

		for each_dict in anchor_positives:
			for k, v in each_dict.items():
				a_path.append(self.data_folder_path+'/'+ self.glob_iter[k]+'/'+v[0])
				p_path.append(self.data_folder_path+'/'+ self.glob_iter[k]+'/'+v[1])

		# print(len(a_path), len(n_path))
		for map_dict in neg_img_list:
			for k, v in map_dict.items():
				n_path.append(self.data_folder_path+'/'+ self.glob_iter[k]+'/'+v)


		anchors = self._paths2tensor(a_path)
		positives = self._paths2tensor(p_path)
		negatives = self._paths2tensor(n_path)

		return anchors, positives, negatives


class CustomeTripletLoss(torch.nn.Module):
	def __init__(self, 
		device: Union[int, torch.device], 
		margin:int = 0.9, 
		batch_size:int = 2, 
		return_examples:int = 80,
		p_n_ratio:int = 4
		)->None:
		super(CustomeTripletLoss, self).__init__()
		self.margin = torch.tensor([margin]*batch_size, dtype = torch.float32).to(device)
		self.zero = torch.zeros(self.margin.shape, dtype = torch.float32).to(device)
		self.batch_size = batch_size
		self.return_examples = return_examples
		self.p_n_ratio = p_n_ratio
		return None

	
	def get_cosim(self, input1:torch.Tensor, input2:torch.Tensor)->torch.Tensor:
		norm_1 = torch.unsqueeze(torch.norm(input1, dim = -1), dim = -1)
		norm_2 = torch.unsqueeze(torch.norm(input2, dim = -1), dim = 1)
		
		length_mul_matrix = 1/torch.mul(norm_1, norm_2)
		
		dot_product = torch.matmul(input1, torch.transpose(input2,1,-1))

		dot_product_score = torch.mul(dot_product, length_mul_matrix)

		return 1.0 - torch.mean(dot_product_score, dim = (1,2))

	def forward(self, 
				a_embeddings: torch.Tensor, 
				p_embeddings: torch.Tensor, 
				n_embeddings: torch.Tensor,
				)->torch.Tensor:
		"""
		Parameters:
			a_embeddings: torch.Tensor shape (self.return_examples, embedding_size (e.g 512) )
			p_embeddings: torch.Tensor shape (self.return_examples, embedding_size (e.g 512) )
			n_embeddings: torch.Tensor shape (self.return_examples*self.p_n_ratio, embedding_size (e.g 512) )
		
		"""
		a_embeddings = a_embeddings.reshape(self.batch_size, self.return_examples, 512)
		p_embeddings = p_embeddings.reshape(self.batch_size, self.return_examples, 512)
		n_embeddings = n_embeddings.reshape(self.batch_size, self.return_examples*self.p_n_ratio, 512)

		sim_a_p = self.get_cosim(a_embeddings, p_embeddings)
		sim_a_n = self.get_cosim(a_embeddings, n_embeddings)

		tripletloss = sim_a_p - sim_a_n + self.margin
		return torch.sum(torch.max(tripletloss,self.zero))