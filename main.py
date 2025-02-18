from PatchEmbedding import PatchEmbedding
from SimilaritySearchPatchGrids import SimilarityByGrids
from CreatePatches import ImagePatching
from tissueRegionCalculator import tissueRegionization

import os
import shutil
import matplotlib.pyplot as plt
from huggingface_hub import from_pretrained_keras

OPENSLIDE_PATH = 'C:/openslide-bin-4.0.0.6-windows-x64/bin'

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

#### Change these variables according to need
# Size of chunks of higher level in number of patches (CHUNKS_COUNT by CHUNKS_COUNT number of patches)
CHUNKS_COUNT = 4
PATCH_PIXELS = (224, 224)                   # Size in pixels of each lower level patches
EMPTY_REGIONS_THRESHOLD = 20                # Percentage range + and - allowed for non-tissue regions of similar images


def create_new_level(tumorfile, level, highreslevel):
    
    try:
        img_path = f'tumors/{tumorfile}'
    except Exception as e:
        print(f"{e}: Tumor file {img_path} does not exist in tumors/")
    
    img = openslide.OpenSlide(img_path)
    new_level_size = int((PATCH_PIXELS[0] * CHUNKS_COUNT) / (img.level_downsamples[level] / img.level_downsamples[highreslevel]))
    
    patch = ImagePatching(tumorfile)
    print(f"\nCreating Level {level} Patches...\n")
    patch.create_level_patch(img, level, (new_level_size, new_level_size))


def main():
    level = None
    highreslevel = 1
    tumorfile = str(input("\nInput file name of tumor tif file in tumors folder to use: "))
    
    if os.path.exists('tumors') and os.path.isdir('tumors'):
        pass
    else:
        os.mkdir('tumors')
    
    cont = str(input("\nExtract patches for a new tumor image (MUST do for a new tumor file) (y/n)? "))
    
    if cont.lower() == 'y':
        cont = str(input("Are you sure you want to proceed (y/n)? "))
    
    if cont.lower() == 'y':
        patch = ImagePatching(tumorfile)
        level = patch.compile_patch_folders(CHUNKS_COUNT)
    
    if level is None:
        level = int(input("\nEnter Patches level to convert level 1 patch embeddings to: "))
    
    if os.path.exists(f'Level {level} Patches') and os.path.isdir(f'Level {level} Patches'):
        print(f"\nFolder 'Level {level} Patches' already exists.")
        cntr = str(input("Replace folder (y/n)? "))
        print("\n")
        
        if cntr.lower() == 'y':
            shutil.rmtree(f'Level {level} Patches')
            
            print(f"Folder '{f'Level {level} Patches'}' is being replaced.")
            print(f"\nCreating 'Level {level} Patches' folder...")
            create_new_level(tumorfile, level, highreslevel)
        else:
            pass
    else:
        print(f"\nCreating 'Level {level} Patches' folder...")
        create_new_level(tumorfile, level, highreslevel)
    
    # Main patches of WSI for conversion
    lowres_imgs = f'Level {level} Patches/'
    highres_imgs = f'Level {highreslevel} Patches/'
    
    # Importing Path Foundation model
    pf_model = from_pretrained_keras('google/path-foundation')
    
    # Creating object for manipulation
    embedding_obj = PatchEmbedding(
            highres_patches_folder=highres_imgs,
            highreslevel=highreslevel,
            lowres_patches_folder=lowres_imgs,
            lowreslevel=level,
            model=pf_model
    )
    
    try:
        img_path = f'tumors/{tumorfile}'
    except Exception as e:
        print(f"{e}: Tumor file {img_path} does not exist in tumors/")
    
    img = openslide.OpenSlide(img_path)
    
    # Creating averaged embeddings
    embedding_obj.compile_new_embeddings(level_dimensions=img.level_dimensions[highreslevel], imgs_sz=PATCH_PIXELS, patch_size=CHUNKS_COUNT)
    
    # Performing Similarity Search based on index of patch in higher level folder
    avg_embeddings_path = os.path.join("embeddings", f"averaged_embeds_level_{level}.pickle")
    imgs_folder = f"Level {level} Patches"
    
    patches = SimilarityByGrids(avg_embeddings_path, imgs_folder)
    tissuecalc = tissueRegionization(EMPTY_REGIONS_THRESHOLD)
    
    imgs_paths = []
    
    # Getting list of paths to all patches' images
    for file in os.listdir(imgs_folder):
        imgs_paths.append(os.path.join(imgs_folder, file))
    
    # Calculating tissue regions and saving
    tissue_percents = tissuecalc.calculateEmptyPercentage(imgs_paths)
    
    index = input(f"\n\nEnter index of image file in 'Level {level} Patches' to be used for similarity search (Enter nothing to stop): ")
    
    while index.isdigit():
        patches.showSimilarities(int(index), tissue_percents)
        index = input(f"\n\nEnter index of image file in 'Level {level} Patches to be used for similarity search: ")


if __name__ == '__main__':
    main()