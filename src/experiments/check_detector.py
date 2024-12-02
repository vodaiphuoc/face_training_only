"""This file is only for local development """
from src.server.Inference import EmbeddingModel
from src.firebase import Firebase_Handler
from src.mongodb import Mongo_Handler
from src.server.cookie_handler import SessionCookie
from src.utils import get_program_config
import cv2 as cv
import glob
from tqdm import tqdm
from mtcnn import MTCNN
import uuid
import numpy as np

master_config = get_program_config()
model = EmbeddingModel(use_lite_model= False,
                       detector_model_name= master_config['detetor_name'], 
                       reg_model_name= 'Facenet512', #master_config['reg_model_name'],
                       running_mode= 'init_push'
                       )

user_folders = glob.glob("face_dataset\images\*")

total_count = 0
fail_count = 0
master_init_data = {}
for user_folder in user_folders:
    user_name = user_folder.split('\\')[-1]

    user_imgs = []
    user_paths = []
    for path in glob.glob(f"face_dataset\images\{user_name}\*"):
        
        image = cv.imread(path)
        origin_size = image.shape
        # resize_img = cv.resize(image, dsize= (960, 1280))
        # print(f'original size: {origin_size}, new size: {resize_img.shape}')
        user_imgs.append(image)
        total_count+=1
    
    # infernce in batch
    try:
        embedding_result = model.forward(input_image= user_imgs)
        if isinstance(embedding_result, bool):
            fail_count += 1
            print(user_name)
            continue
        else:
            embeddings, is_batch, total_faces = embedding_result
            
            assert is_batch, f"Must be batch"
            assert embeddings.ndim == 2 , f"Found {embeddings.ndim} dims"
            assert embeddings.shape[-1] == 128 or embeddings.shape[-1] == 512, \
                f"Found {embeddings.shape[-1]}"
            

            master_init_data[user_name] = embeddings

            for face in total_faces:
                cv.imwrite('face_dataset/temp/temp_'+ str(uuid.uuid4())+ user_name+'.jpg', face)

    except Exception as e:
        print("Error for user:",user_name, f" with {e} error")

print(total_count)
print(fail_count)


sim_result = {}
for user_name, embedding in master_init_data.items():
    left_user_list = [ele for ele in master_init_data.keys() if ele != user_name]

    inner_group_score = np.dot(embedding, embedding.transpose())
    inner_group_score = inner_group_score[~np.eye(inner_group_score.shape[0],dtype=bool)].reshape(inner_group_score.shape[0],-1)

    other_embeddings = np.concat([master_init_data[other_user] 
                                 for other_user in left_user_list], 
                                 axis= 0)
    outer_group_score = np.dot(embedding, other_embeddings.transpose())
    outer_group_score = np.max(outer_group_score, axis= -1)

    assert outer_group_score.ndim == 1

    sim_result[user_name] = [ True if np.max(inner_score) > outer_score else False
        for inner_score, outer_score in zip(inner_group_score, outer_group_score)
    ]

print(sim_result)