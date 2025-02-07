import os
import shutil

OPENSLIDE_PATH = 'C:/openslide-bin-4.0.0.6-windows-x64/bin'

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


class ImagePatching:
    def __init__(self, tumorfile):
        self.tumorfile = tumorfile
    
    def compile_patch_folders(self, chunks_size):
        
        if os.path.exists('tumors') and os.path.isdir('tumors'):
            pass
        else:
            print("\nCreating tumors folder...")
            os.mkdir('tumors')
            print("Created tumors folder")
            print("\nStopping program as tumors folder is empty.")
            return
        
        try:
            img_path = f'tumors/{self.tumorfile}'
        except Exception as e:
            print(f"{e}: Tumor file {img_path} does not exist in tumors/")
        
        level = int(input("Enter Patches level to convert level 1 patch embeddings to: "))
        
        img = openslide.OpenSlide(img_path)
        
        print("\nCreating Level 1 Patches...\n")
        level1_downsamples = self.create_level_patch(img)
        new_patch_size = int((224 * chunks_size) / (img.level_downsamples[level] / level1_downsamples))     # Size in pixels of new level's patches
        
        print(f"\nCreating Level {level} Patches...\n")
        self.create_level_patch(img, level, (new_patch_size, new_patch_size))
        
        return level
    
    
    def create_level_patch(self, img, level=1, size_increments=(224, 224)):

        start_coords = (0, 0)
        end_coords = img.level_dimensions[level]

        coords_multiply = int(img.level_downsamples[level])
        
        LevelFolder = f'Level {level} Patches'

        if os.path.exists(LevelFolder) and os.path.isdir(LevelFolder):
            print(f"Folder '{LevelFolder}' already exists.")
            cntr = str(input("Replace folder (y/n)? "))
            if cntr.lower() == 'y':
                shutil.rmtree(LevelFolder)
                print(f"Folder '{LevelFolder}' is being replaced.")
            else:
                return img.level_downsamples[level]
        else:
            print(f"Folder '{LevelFolder}' does not exist.")
        
        try:
            os.makedirs(LevelFolder)
        except Exception as e:
            raise(f"{e}: Folder could not be created or replaced.")

        row = 0

        # Creating patches and putting in Folder replaced or created
        for y in range(start_coords[1], end_coords[1], size_increments[1]):
            column = 0
            
            for x in range(start_coords[0], end_coords[0], size_increments[0]):
                x_mult = x * coords_multiply
                y_mult = y * coords_multiply
                
                patch_img = img.read_region((x_mult, y_mult), level, size_increments).convert("RGB")        # Extracting a patch of the image
                
                patch_img.save(f"{LevelFolder}/tumor_patch_{row:04d}_by_{column:04d}.jpeg", format="JPEG")  # Saving patch of the image in folder as jpeg file
                
                column += 1
            
            print(f"Finished patch coords {row:04d} by {column:04d}")
            
            row += 1
        
        return img.level_downsamples[level]