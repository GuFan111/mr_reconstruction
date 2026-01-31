import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import glob

def preprocess_dicom_to_npy(src_root, save_root, target_res=(128, 128, 128)):
    os.makedirs(save_root, exist_ok=True)
    # 获取所有子文件夹 (MR1xT, MR2xT...)
    sub_dirs = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]

    for sub_dir in sub_dirs:
        path = os.path.join(src_root, sub_dir)
        print(f"Processing {sub_dir}...")

        # 使用 SimpleITK 读取 DICOM 序列
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        itk_img = reader.Execute()

        # 转换为 numpy
        img_array = sitk.GetArrayFromImage(itk_img) # [D, H, W]
        img_array = img_array.transpose(2, 1, 0)    # 转为 [X, Y, Z]

        # 归一化 (MR 图像通常需要基于分位数或最大值)
        img_array = img_array.astype(np.float32)
        p99 = np.percentile(img_array, 99.5)
        img_array = np.clip(img_array / p99, 0, 1)

        # 重采样/缩放到 target_res
        # 利用 SimpleITK 的重采样功能或 scipy.ndimage.zoom
        import scipy.ndimage
        factors = [t/s for t, s in zip(target_res, img_array.shape)]
        img_resampled = scipy.ndimage.zoom(img_array, factors, order=1)

        # 保存为模型可读的格式
        save_name = f"{sub_dir}.npy"
        np.save(os.path.join(save_root, save_name), img_resampled)

if __name__ == "__main__":
    src = "D:\CODE\DaChuang\MR_Reconstruction\data\dataset\data_src"
    dst = "D:\CODE\DaChuang\MR_Reconstruction\data\dataset\processed_npy"
    preprocess_dicom_to_npy(src, dst)