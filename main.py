from PatchEmbedding import PatchEmbedding
from SimilaritySearchPatchGrids import SimilarityByGrids
from CreatePatches import ImagePatching

import os
import shutil
from huggingface_hub import from_pretrained_keras

OPENSLIDE_PATH = 'C:/openslide-bin-4.0.0.6-windows-x64/bin'

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

#### Change these variables according to need
# Size of chunks of higher level in number of patches (CHUNKS_SIZE by CHUNKS_SIZE number of patches)
CHUNKS_SIZE = 10


def create_new_level(level):
    print(f"\nCreating 'Level {level} Patches' folder...")
    tumorfile = str(input("\nInput file name of tumor tif file in tumors folder to use: "))
    
    try:
        img_path = f'tumors/{tumorfile}'
    except Exception as e:
        print(f"{e}: Tumor file {img_path} does not exist in tumors/")
    
    img = openslide.OpenSlide(img_path)
    new_level_size = int((224 * CHUNKS_SIZE) / (img.level_downsamples[level] / img.level_downsamples[1]))
    
    patch = ImagePatching()
    print(f"\nCreating Level {level} Patches...\n")
    patch.create_level_patch(img, level, (new_level_size, new_level_size))


def main():
    level = None
    
    if os.path.exists('tumors') and os.path.isdir('tumors'):
        pass
    else:
        os.mkdir('tumors')
    
    cont = str(input("\nExtract patches for a new tumor image (MUST do for a new tumor file) (y/n)? "))
    
    if cont.lower() == 'y':
        cont = str(input("Are you sure you want to proceed (y/n)? "))
    
    if cont.lower() == 'y':
        patch = ImagePatching()
        level = patch.compile_patch_folders(CHUNKS_SIZE)
    
    if level is None:
        level = int(input("\nEnter Patches level to convert level 1 patch embeddings to: "))
    
    if os.path.exists(f'Level {level} Patches') and os.path.isdir(f'Level {level} Patches'):
        print(f"\nFolder 'Level {level} Patches' already exists.")
        cntr = str(input("Replace folder (y/n)? "))
        
        if cntr.lower() == 'y':
            shutil.rmtree(f'Level {level} Patches')
            
            print(f"Folder '{f'Level {level} Patches'}' is being replaced.")
            create_new_level(level)
        else:
            pass
    else:
        create_new_level(level)
    
    # Main patches of WSI for conversion
    lowres_imgs = f'Level {level} Patches/'
    highres_imgs = 'Level 1 Patches/'
    
    # Importing Path Foundation model
    pf_model = from_pretrained_keras('google/path-foundation')
    
    # Creating object for manipulation
    embedding_obj = PatchEmbedding(
            highres_patches_folder=highres_imgs,
            highreslevel=1,
            lowres_patches_folder=lowres_imgs,
            lowreslevel=level,
            model=pf_model
    )
    
    # Creating averaged embeddings
    embedding_obj.compile_new_embeddings()
    
    # Performing Similarity Search based on index of patch in higher level folder
    avg_embeddings_path = os.path.join("embeddings", f"averaged_embeds_level_{level}.pickle")
    imgs_folder = f"Level {level} Patches/"
    
    patches = SimilarityByGrids(avg_embeddings_path, imgs_folder)
    
    index = int(input(f"\n\nEnter index of image file in 'Level {level} Patches to be used for similarity search: "))
    patches.showSimilarities(index)


if __name__ == '__main__':
    main()