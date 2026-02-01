import SimpleITK as sitk

file_path = r"D:\CODE\DaChuang\MR_Reconstruction\data\dataset\data_src\MR1xT\0034895784_MR1xT_image00000.DCM"
reader = sitk.ImageFileReader()
reader.SetFileName(file_path)
reader.ReadImageInformation()

# 获取空间信息
print(f"Origin: {reader.GetOrigin()}")
print(f"Spacing: {reader.GetSpacing()}")
print(f"Direction: {reader.GetDirection()}")

# 查看特定的 DICOM 标签内容（例如 0008|0060 为 Modality）
print(f"设备类型: {reader.GetMetaData('0008|0060')}")