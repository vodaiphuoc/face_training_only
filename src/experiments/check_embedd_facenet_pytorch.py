from src.server.Facenet_pytorch.utils.fine_tuning_utils import fixed_image_standardization
from src.server.Facenet_pytorch.inception_resnet_v1 import InceptionResnetV1

import cv2 as cv
from typing import List, Literal
import numpy as np
import glob
import torch
import uuid
from tqdm import tqdm
import re
import os
import json
import time


# @torch.compile(fullgraph = True, dynamic = False, mode = 'reduce-overhead')
def get_cosim(input1:torch.Tensor, input2:torch.Tensor)->float:
	"""
	input1,input2: torch.Tensor shape (B_{i}, 512)
	"""
	input2 = torch.cat([input2, input2, input2, input2, input2],dim = 0)
	norm_1 = torch.unsqueeze(torch.norm(input1, dim =1), dim = 1) 
	norm_2 = torch.unsqueeze(torch.norm(input2, dim =1), dim = 0)
	length_mul_matrix = 1/torch.mul(norm_1, norm_2)

	dot_product = torch.matmul(input1, torch.transpose(input2,0,1))

	dot_product_score = torch.mul(dot_product, length_mul_matrix)

	return torch.max(dot_product_score)


class Test_Embeddings(object):
	def __init__(self,
				data_folder_path: str,
				pretrained_weight_dir: str,
				model_string: Literal['casia-webface','fine_tuning'],
				json_path:str = None,
				users_from_json: bool = False,
				):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.recognition_model = InceptionResnetV1(pretrained = model_string, 
													classify=False, 
													num_classes=None, 
													dropout_prob=0.6,
													device=self.device,
													pretrained_weight_dir = pretrained_weight_dir
	    									).to(self.device)
		self.recognition_model.eval()
		self.data_folder_path = data_folder_path
		self.users_from_json = users_from_json
		self.json_path = json_path

	def _run_single_user(self, 
				user_name:str,
				):
		user_imgs = []
		for path in glob.glob(f"{self.data_folder_path}/{user_name}/*"):
			image = cv.imread(path)
			assert image is not None, f"{path}"
			image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
			user_imgs.append(image)
		
		stack_faces = torch.tensor(np.stack(user_imgs)).permute(0,3,1,2)
		assert stack_faces.shape[0] == len(user_imgs) and stack_faces.shape[1] == 3
		stack_faces =fixed_image_standardization(stack_faces).to(self.device)
		with torch.no_grad():
			embeddings = self.recognition_model(stack_faces)
		# assert embeddings.shape[0] == len(filtered_faces)
		assert embeddings.shape[1] == 512
	
		return {'user_name': user_name,
				'password': '12345',
				'embeddings': embeddings
				}

	def _get_total_init_user_data(self):
		if self.users_from_json:
			assert self.json_path is not None
			with open(self.json_path,'r') as f:
				user_folders_list = json.load(f)

				self.user_folders = [
					self.data_folder_path +'/'+ list(ele.values())[0] 
					for ele in user_folders_list
				]
		else:
			self.user_folders = glob.glob(f"{self.data_folder_path}/*")

		master_init_data = []

		for user_folder in tqdm(self.user_folders, total = len(self.user_folders)):
			user_name = os.path.split(user_folder)[-1].split('.')[0]
			user_init_data = self._run_single_user(user_name = user_name)
			master_init_data.append(user_init_data)
		
		return master_init_data
	

	def pipelines(self):
		master_init_data = self._get_total_init_user_data()
				
		result = {}
		for main_user_dir in glob.glob(f"{self.data_folder_path}/*_*"):
			user_name = os.path.split(main_user_dir)[-1].split('.')[0]
			user_embeddings_dict = self._run_single_user(user_name = user_name)

			user_accuracy = 0
			result[user_name] = {}

			pred_user_list = []
			# loop through each embedding
			for embedding in user_embeddings_dict['embeddings']:
				embedding = torch.unsqueeze(embedding, dim = 0)

				# compute score over all training data
				score_dict = {
					user_dict['user_name']: get_cosim(user_dict['embeddings'], embedding).item()
					for user_dict in master_init_data
				}
				# sort score dict in ascending order
				score_dict = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1])}

				pred_name = list(score_dict.keys())[-1]
				if pred_name == user_name:
					user_accuracy += 1
				pred_user_list.append(pred_name)

			user_accuracy = user_accuracy/len(user_embeddings_dict['embeddings'])

			result[user_name]['mean_accuracy'] = user_accuracy
			result[user_name]['pred_user_list'] = pred_user_list

		print(result)

		total_mean_acc = 0.0
		for k,v in result.items():
			total_mean_acc += v['mean_accuracy']

		print('Total mean accuracy: ', total_mean_acc/len(result))