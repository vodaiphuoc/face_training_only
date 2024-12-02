"""This file is only for local development """
from src.server.Inference import EmbeddingModel
from src.firebase import Firebase_Handler
from src.mongodb import Mongo_Handler
from src.server.cookie_handler import SessionCookie
from src.utils import get_program_config
import cv2 as cv
import glob
from tqdm import tqdm
import numpy as np
import itertools
import uuid

master_config = get_program_config()
model = EmbeddingModel(use_lite_model= False,
                       detector_model_name= master_config['detetor_name'], 
                       reg_model_name= master_config['reg_model_name'],
                       running_mode= 'init_push'
                       )

user_folders = glob.glob("face_dataset\images\*")
master_init_data = []
for user_folder in user_folders:
    user_name = user_folder.split('\\')[-1]

    user_imgs = []
    for path in glob.glob(f"face_dataset\images\{user_name}\*"):
        image = cv.imread(path)
        user_imgs.append(image)
    
    # infernce in batch
    embedding_result = model.forward(input_image = user_imgs)
    if isinstance(embedding_result, bool):
        continue
    else:
        embeddings, is_batch, total_faces = embedding_result
        assert is_batch
        assert embeddings.ndim == 2
        assert embeddings.shape[-1] == 128 or embeddings.shape[-1] == 512, f"found {embeddings.shape[-1]}"
        
        user_init_data = [{
                'user_name': user_name,
                'password': '123',
                'embedding': embeddings[id].tolist(),
        } 
        for id in range(embeddings.shape[0])
        ]
        master_init_data.extend(user_init_data)


engine = Mongo_Handler(master_config= master_config, 
                        ini_push= True,
                        init_data= master_init_data)

cookie_ssesion = SessionCookie(cookie_name = "app_cookie",
                                secret_key = "DONOTUSE",
                                db_handler = engine)

# for user_folder in user_folders:
#     cookie_ssesion.make_new_session(action= 'signup')


check_sim_result = {}
for user_folder in user_folders:
    user_name = user_folder.split('\\')[-1]
    print(user_name)

    user_imgs = []
    for path in glob.glob(f"face_dataset\images\{user_name}\*"):
        image = cv.imread(path)
        user_imgs.append(image)
    
    # infernce in batch
    embedding_result = model.forward(input_image = user_imgs)
    if isinstance(embedding_result, bool):
        continue
    else:
        embeddings, is_batch, total_faces = embedding_result

        for face in total_faces:
                cv.imwrite('face_dataset/temp/temp_'+ str(uuid.uuid4())+ user_name+'.jpg', face)

        for ith_embed, ith_next_embed in itertools.pairwise(range(len(embeddings))):
            query_embed = np.stack([embeddings[ith_embed],
                                     embeddings[ith_next_embed]], 
                                     axis = 0)
            
            # assert query_embed.ndim == 2 and query_embed.shape[-1] == 512 and query_embed.shape[0] == 2
            pred_user_name = engine.searchUserWithEmbeddings(batch_query_embeddings= query_embed)

            if check_sim_result.get(user_name) is None:
                check_sim_result[user_name] = []

            check_sim_result[user_name].append(pred_user_name)

print(check_sim_result)