import numpy as np 
import os
import pickle

path = os.path.join("embeddings", f"averaged_embeds_level_6.pickle")

with open(path, 'rb') as file:
       embeds = pickle.load(file)

# Determine the correct embedding size from valid embeddings
valid_embeddings = [d['embedding'] for d in embeds if isinstance(d['embedding'], np.ndarray) and not np.isnan(d['embedding']).any()]
embedding_size = len(valid_embeddings[0]) if valid_embeddings else 0  # Assuming all valid embeddings have the same size

# Replace NaN embeddings with zero-filled arrays
for d in embeds:
       if isinstance(d['embedding'], (float, np.float64)) and np.isnan(d['embedding']):
              d['embedding'] = np.zeros(embedding_size)

for i in embeds:
       print(i, end="\n")

print(len(embeds))