import os
from glob import glob
import shutil

raw_path = "../data/raw/phieu_chi_tat_toan_so_tiet_kiem/"
gt_path = "../data/gt/phieu_chi_tat_toan_so_tiet_kiem/"
extension = "_ocrfile"


os.makedirs("dataset/image", exist_ok = True)
os.makedirs("dataset/json", exist_ok = True)
list_file = os.listdir(gt_path)


for i in range(len(list_file)):
    print(list_file[i])
    shutil.copy(os.path.join(gt_path, list_file[i]), f"dataset/json/{i}.json")
    # print( os.path.join(raw_path, list_file[i].replace(".json", extension), "0.png"))
    image_file = os.path.join(raw_path, list_file[i].replace(".json", extension), "0.png")
    shutil.copy(image_file, f"dataset/image/{i}.png")