# self.detector = MTCNN(image_size=160, 
        #             margin=0, 
        #             min_face_size=20,
        #             thresholds=[0.6, 0.7, 0.7], 
        #             factor=0.709, 
        #             post_process=True,
        #             select_largest=True, 
        #             selection_method=None, 
        #             keep_all=True,
        #             device=self.device,
        #             p_state_dict_path = p_state_dict_path,
        #             r_state_dict_path = r_state_dict_path,
        #             o_state_dict_path = o_state_dict_path,
        #             )
from src.server.models.Facenet_pytorch.mtcnn import MTCNN, fixed_image_standardization
from src.server.models.Facenet_pytorch.inception_resnet_v1 import InceptionResnetV1
from src.mongodb import Mongo_Handler
from src.utils import get_program_config
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


@torch.compile(fullgraph = True, dynamic = False, mode = 'reduce-overhead')
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
	

	def pipelines(self, 
				run_init_push: bool, 
				evaluation: bool
				):
		master_config = get_program_config()
		master_init_data = self._get_total_init_user_data()
		# print('number init data: ',len(master_init_data))

		if run_init_push:
			db_engine = Mongo_Handler(master_config= master_config,
						ini_push= True,
						init_data= master_init_data)
		
		if evaluation:
			# db_engine = Mongo_Handler(master_config= master_config,
			# 			ini_push= False)

			result = {}
			for main_user_dir in glob.glob(f"{self.data_folder_path}/*_*"):
				user_name = os.path.split(main_user_dir)[-1].split('.')[0]
				user_embeddings_dict = self._run_single_user(user_name = user_name)

				result[user_name] = db_engine.searchUserWithEmbeddings_V2(user_embeddings_dict['embeddings'].cpu())

			print(result)