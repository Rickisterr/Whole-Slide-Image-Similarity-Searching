import os
import numpy as np
import tensorflow as tf
import shutil
from sklearn.cluster import DBSCAN
from collections import Counter
import pickle

OPENSLIDE_PATH = 'C:/openslide-bin-4.0.0.6-windows-x64/bin'
EPS = 0.15
MIN_SAMPLES = 10

import os
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import matplotlib.pyplot as plt
from PIL import Image


class PatchEmbedding:
    def __init__(self, highres_patches_folder, highreslevel, lowres_patches_folder, lowreslevel, model):
        
        self.highres_imgs_path = highres_patches_folder
        self.highres_level = highreslevel
        self.lowres_imgs_path = lowres_patches_folder
        self.lowres_level = lowreslevel
        
        self.embedding_model = model
        
    
    def _create_embeddings(self, imgs_path, level):
        pickle_path = os.path.join('embeddings', f'embeds_level_{level}.pickle')
        
        if os.path.exists(pickle_path) and os.path.isfile(pickle_path):
            print(f"File '{pickle_path}' already exists.")
            return
        
        i = 0
        embeddings = []

        for file_name in os.listdir(imgs_path):
            file_path = os.path.join(imgs_path, file_name)                                      # Creating full path to a specific file
            
            if os.path.isfile(file_path):
                print(f"File {i}: Processing <{file_path}>")
                
                image = Image.open(file_path).convert("RGB")                                    # Getting RGB values of PIL Image object
                
                image = tf.cast(tf.expand_dims(np.array(image), axis=0), tf.float32) / 255.0    # Turning image into a tensor
                
                infer = self.embedding_model.signatures["serving_default"]                      # Default training process
                embed = infer(tf.constant(image))                                               # Inferring image embeddings
                
                embed = embed['output_0'].numpy().flatten()                                     # Getting final embedding of output
                
                embeddings.append({
                    "ID": len(embeddings),
                    "filepath": file_path,
                    "embedding": embed
                })                                                                              # Appending new embedding to array of all embeddings
                
                print(f"Number of Embeddings: {len(embeddings)}", end='\n')
                
                i += 1

        print("\nSaving embeddings and their paths")
        
        # TODO: Change these save methods for embeddings to sqlite3
        # Saving numpy array of embeddings into embeds.pickle        
        with open(pickle_path, 'wb') as file:
            pickle.dump(embeddings, file)
            print(f"Embeddings with 'ID', 'filepath', and 'embedding' saved at {pickle_path}.")
        
        return
    
    
    def _combine_patch_embeddings(self, level_dimensions=(48896, 109824), imgs_sz=(224, 224), patch_size=10):
        embeds_file = f'embeds_level_{self.highres_level}.pickle'
        embeds_path = os.path.join('embeddings', embeds_file)
        
        try:
            self._create_embeddings(self.highres_imgs_path, self.highres_level)
        except Exception as e:
            raise(f"{e}: Folder could not be created or replaced.")
        
        with open(embeds_path, 'rb') as file:
            embeddings = pickle.load(file)
            
        print(f"{len(embeddings)} embeddings retrieved from {embeds_path}.")
        print()
        print()
        
        cols = int(np.ceil(level_dimensions[0] / imgs_sz[0]))
        rows = int(np.ceil(level_dimensions[1] / imgs_sz[1]))
        
        file_path_to_index = {item["filepath"]: idx for idx, item in enumerate(embeddings)}
        
        patches_grids = []
        
        for idx in range(0, rows, patch_size):
            for jdx in range(0, cols, patch_size):
                single_grid = []
                
                for offset_idx in range(idx, idx+patch_size):
                    for offset_jdx in range(jdx, jdx+patch_size):
                        # Path to single patch considered
                        patch_path = os.path.join(self.highres_imgs_path, f'tumor_patch_{offset_idx:04d}_by_{offset_jdx:04d}.jpeg')
                        
                        # If path exists as a file, getting its index and finding its equivalent embedding
                        index = file_path_to_index.get(patch_path, -1)
                        embed = embeddings[index].get("embedding")
                        
                        single_grid.append(embed)
                
                # TODO: Check other hyperparameters for better results
                dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
                labels = dbscan.fit_predict(single_grid)

                # Finding the majority cluster
                cluster_counts = Counter(labels)
                majority_cluster = max(cluster_counts, key=cluster_counts.get)
                
                majority_embeddings = []
                
                for i in labels:
                    if labels[i] == majority_cluster:
                        majority_embeddings.append(single_grid[i])

                # Computing the average embedding of the majority cluster
                average_embedding = np.mean(majority_embeddings, axis=0)
                
                patches_grids.append({
                    "new_patch_coords": f"{int(idx/patch_size):04d}_by_{int(jdx/patch_size):04d}",
                    "embedding": average_embedding
                })
                
                print(f"Embedding of grid {idx}x{jdx} processed as new level's {int(idx/patch_size):04d}x{int(jdx/patch_size):04d} patch.")
        
        print()
        
        return patches_grids
    
    
    def compile_new_embeddings(self):
        pickle_path = os.path.join('embeddings', f'averaged_embeds_level_{self.lowres_level}.pickle')
        
        if os.path.exists(pickle_path) and os.path.isfile(pickle_path):
            print(f"\nFile '{pickle_path}' already exists.")
            cont = str(input("Replace file and continue (y/n)? "))
            
            if cont.lower() == 'y':
                os.remove(pickle_path)
                print("File being replaced.")
            else:
                return
        
        patches_grids = self._combine_patch_embeddings()
        
        # Saving embeddings of patches grids as pickle file
        with open(pickle_path, 'wb') as file:
            pickle.dump(patches_grids, file)
            print(f"Embeddings with 'ID', 'filepath', and 'embedding' saved at {pickle_path}.\n")
        
        return