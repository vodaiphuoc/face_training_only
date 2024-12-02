# from src.server.models.Facenet_pytorch.fine_tuning import FineTuner
from src.experiments.check_embedd_facenet_pytorch import Test_Embeddings
import torch
from collections import OrderedDict
import json
import glob


if __name__ == '__main__':
	Test_Embeddings(data_folder_path = 'face_dataset/faces_only', 
					users_from_json = True,
					json_path = 'face_dataset/dataset.json',
					pretrained_weight_dir = 'src\\server\\models\\pretrained_weights\\Facenet_pytorch',
					model_string = 'fine_tuning'
				).pipelines(run_init_push = False, 
				evaluation = True)

	