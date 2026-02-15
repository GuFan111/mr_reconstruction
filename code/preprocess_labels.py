import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from tqdm import tqdm
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    Orientationd, 
    Spacingd, 
    Resized, 
    EnsureTyped
)

def preprocess_labels(src_root, save_root, target_res=(256, 256, 128)):
    os.makedirs(save_root, exist_ok=True)
    
    # 定义 Label 专用流水线
    # 关键点：mode='nearest'，防止标签插值产生小数
    label_pipeline = Compose([
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
        Orientationd(keys=["label"], axcodes="RAS"), # 必须与 Image 一致
        Spacingd(keys=["label"], pixdim=(1.5, 1.5, 1.5), mode="nearest"), # 必须与 Image 一致
        Resized(keys=["label"], spatial_size=target_res, mode="nearest"),
        EnsureTyped(keys=["label"])
    ])

    # AMOS 的 Label 通常在 labelsTr 和 labelsVa 文件夹
    folders = ['labelsTr', 'labelsVa']
    
    for folder in folders:
        folder_path = os.path.join(src_root, folder)
        if not os.path.exists(folder_path): continue
        
        print(f"--- Processing Labels in {folder} ---")
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.nii.gz')])
        
        for f in tqdm(files):
            try:
                # 过滤：确保只处理 MRI (ID >= 500)
                idx = int(f.split('_')[1].split('.')[0])
                if idx < 500: continue 
                
                img_path = os.path.join(folder_path, f)
                output = label_pipeline({"label": img_path})
                
                # 转为 uint8 或 int8 节省空间 (类别数通常 < 255)
                lbl_array = output["label"].squeeze().numpy().astype(np.uint8)
                
                # 保存为 .npy
                save_name = f"amos_{idx}_label.npy"
                np.save(os.path.join(save_root, save_name), lbl_array)
                
            except Exception as e:
                print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    src = "/root/autodl-tmp/Proj/data/amos22"
    dst = "/root/autodl-tmp/Proj/data/amos_mri_label_npy" # 新的 Label 存储路径
    preprocess_labels(src, dst)