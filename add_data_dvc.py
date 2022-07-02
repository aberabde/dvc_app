import os
from glob import glob


data_dir = ["data_given"]
files = glob(r"data_given/*.csv")
for filePath in files:
        # print(f"dvc add {filePath}")
    os.system(f"dvc add {filePath}")

print("\n #### all files added to dvc ####")