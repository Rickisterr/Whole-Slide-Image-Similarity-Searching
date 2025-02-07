import os
import pickle

embeds_dir = 'embeddings'

files = os.listdir(embeds_dir)

for fdx in range(len(files)):
    print(f"File {fdx}: {files[fdx]}")
print()

files_lis = str(input("Enter the list of indices (space-delimited) for files to display (e.g.: 0 1 2 3...): ")).split(" ")
for fdx in range(len(files_lis)):
    files_lis[fdx] = files[int(files_lis[fdx])]

for file in files_lis:
    path = os.path.join(embeds_dir, file)
    
    # Loading all the data from pickle file onto RAM
    with open(path, "rb") as f:
        data = pickle.load(f)
        
    print(f"\n\nFile: {path}\n\n")
    cntr = str(input("Should embeddings be shown (Warning: Large data) (y/n)? "))
    if cntr.lower() == "y":
        cntr = False
    else:
        cntr = True
    
    print()
    print("Enter number of rows in data to display (start index is 0 and last index is -1)")
    N = input("'x y' for data rows x to y; positive 'x' for rows 0 to x; negative '-y' for rows last-y to last: ").split(" ")
    print()
    
    if len(N) >= 2:
        data = data[int(N[0]):int(N[1])+1]
    else:
        if int(N[0]) >= 0:
            data = data[0:int(N[0])+1]
            
        elif int(N[0]) < 0:
            data = data[int(N[0]):]
        
        
    for idx in range(len(data)):
        for key in list(data[idx].keys()):
            if cntr and (key == 'embedding'):
                continue
            print(f"{key}: {data[idx].get(key)}", end="\t")
        print()
        print()
    
    print("\n")