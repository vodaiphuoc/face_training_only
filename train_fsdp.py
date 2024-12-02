from src.server.models.Facenet_pytorch.fine_tuning import training_loop
from src.experiments.check_embedd_facenet_pytorch import Test_Embeddings
import torch
import torch.multiprocessing as mp

if __name__ == '__main__':
	
	trainer_args = {'num_epochs': 1,
					'gradient_accumulate_steps': 4,
					'lr': 0.0001,
					'pretrained_weight_dir': 'src\\server\\models\\pretrained_weights\\Facenet_pytorch',
					'return_examples': 128,
					'data_folder_path': 'face_dataset/faces_only',
					'ratio_other_user': 0.1,
					'number_celeb_in_train': 500,
					'number_celeb_in_val': 150,
					'batch_size': 4,
					'num_workers': 1
					}

    torch.manual_seed(1)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(training_loop,
        args = (WORLD_SIZE,trainer_args,'fine_tuning.pt'),
        nprocs = WORLD_SIZE,
        join=True)


	# Test_Embeddings(data_folder_path = 'face_dataset/faces_only', 
	# 				pretrained_weight_dir = 'src\\server\\models\\pretrained_weights\\Facenet_pytorch',
	# 				model_string = 'fine_tuning'
	# 			).pipelines(run_init_push = True, evaluation = True)


# !cd /kaggle/working/Checkin_App