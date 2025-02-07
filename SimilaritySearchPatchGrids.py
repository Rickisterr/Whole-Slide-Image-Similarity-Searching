import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image


class SimilarityByGrids:
    def __init__(self, average_embeddings_path, imgs_folder):
        self.images = []
        
        with open(average_embeddings_path, 'rb') as file:
            self.embeds = pickle.load(file)
        
        # Determine the correct embedding size from valid embeddings
        valid_embeddings = [d['embedding'] for d in self.embeds if isinstance(d['embedding'], np.ndarray) and not np.isnan(d['embedding']).any()]
        embedding_size = len(valid_embeddings[0]) if valid_embeddings else 0  # Assuming all valid embeddings have the same size

        # Replace NaN embeddings with zero-filled arrays
        for d in self.embeds:
            if isinstance(d['embedding'], (float, np.float64)) and np.isnan(d['embedding']):
                d['embedding'] = np.full((embedding_size,), 1e-10)
        
        for i in os.listdir(imgs_folder):
            self.images.append(os.path.join(imgs_folder, i))
        
        self.train_embeddings = np.array([np.array(d["embedding"]) for d in self.embeds])
    
    
    
    def similarityCalculations(self, test_embedding_index, k_similarities):
        
        test_embedding = self.train_embeddings[test_embedding_index]
        
        # Normalization of matrices to allow range within -1 to 1
        test_embedding_norm = np.linalg.norm(test_embedding)
        total_norms = np.linalg.norm(self.train_embeddings, axis=1) * test_embedding_norm
        
        # Calculating dot product by dot(A, B) / (norm(A) * norm(B))
        similarities = np.dot(test_embedding, self.train_embeddings.T) / total_norms
        
        indices = np.argsort(similarities)                                              # Ordering indices of similarities in ascending order
        indices = indices[::-1][:k_similarities]                                        # Getting k similarities of highest values
        similar_imgs = [self.images[i] for i in indices]                                # Getting the respective images with most similarity
        similar_percent = [round((similarities[i]*100), 2) for i in indices]         # Getting similarity values of each procured image as a percentage
        
        return similar_imgs, similar_percent


    def showImages(self, test_index, similars, percents, img_size=3):
        main_img_obj = Image.open(self.images[test_index]).convert("RGB")   # Gathering image object from file path
        
        # Showing image in single figure
        num_images = len(similars) + 1
        plt.figure(figsize=(img_size * num_images, img_size))
        
        # Show main image in the first subplot
        plt.subplot(1, num_images, 1)
        plt.imshow(main_img_obj)
        plt.title("Main Test Image")
        plt.axis("off")
        
        # Show similar images
        for i, path in enumerate(similars):
            img_obj = Image.open(path).convert("RGB")  # Gathering image object from file path
            
            plt.subplot(1, num_images, i + 2)
            plt.imshow(img_obj)
            plt.title(f"Similarity: {round(percents[i], 2):.2f}%")
            plt.axis("off")
        
        plt.show()


    def showSimilarities(self, test_index, k=3):
    
        # Calculating similarities and getting similar images
        similars, percents = self.similarityCalculations(test_index, k+1)
        
        # Showing similar images with similarity percentages
        self.showImages(test_index, similars, percents)
        
        return